'use strict';

const VERSION = 2;
const BLOCK_SIZE = 16;
const BUCKET_COUNT = 8;
const DEFAULT_OFF = 2;
const TILE_SIZE = 288;

const FORMAT_PLN1 = 'PLN1';
const FORMAT_PLN2 = 'PLN2';

const COLOR_TRANSFORM = 'rct-u-rg-v-bg-v-minus-u4';
const PREDICTOR = 'med';
const BLOCK_LAYOUT = 'raster-16-cross-row-zero-tail';

const PLN1_NAMES = ['gray'];
const PLN2_NAMES = ['Y', 'U', 'V'];

function rct(r, g, b) {
  const U = r - g;
  const V0 = b - g;
  const Y = g + ((U + V0 + 2) >> 2);
  const V = V0 - (U >> 2);

  return { Y, U, V };
}

function irct(Y, U, V) {
  const V0 = V + (U >> 2);
  const g = Y - ((U + V0 + 2) >> 2);

  return {
    r: U + g,
    g,
    b: V0 + g
  };
}

function wrap8(value) {
  return ((value + 128) & 255) - 128;
}

function wrap9(value) {
  return ((value + 256) & 511) - 256;
}

function zigzag(value) {
  return value >= 0
    ? value * 2
    : -value * 2 - 1;
}

function unzigzag(value) {
  return (value & 1) === 0
    ? value / 2
    : -(value + 1) / 2;
}

function bitWidth(value) {
  return value <= 0 ? 0 : 32 - Math.clz32(value);
}

function validateOff(off) {
  if (!Number.isInteger(off) || off < 0 || off > 31) {
    throw new RangeError('off must be an integer in [0, 31]');
  }
}

function getGeometry(width, height) {
  if (!Number.isInteger(width) || width <= 0) {
    throw new RangeError('width must be a positive integer');
  }

  if (!Number.isInteger(height) || height <= 0) {
    throw new RangeError('height must be a positive integer');
  }

  const size = width * height;

  if (!Number.isSafeInteger(size) || size > 0x7fffffff) {
    throw new RangeError('image is too large');
  }

  const blockCount = Math.ceil(size / BLOCK_SIZE);
  const paddedSymbolCount = blockCount * BLOCK_SIZE;

  return {
    size,
    blockCount,
    paddedSymbolCount,
    tailPadding: paddedSymbolCount - size
  };
}

function checkByteArray(input, label) {
  if (
    !(input instanceof Uint8Array) &&
    !(input instanceof Uint8ClampedArray)
  ) {
    throw new TypeError(
      `${label} must be Uint8Array, Uint8ClampedArray or Buffer`
    );
  }
}

function checkRGB(rgb, size) {
  checkByteArray(rgb, 'RGB data');

  if (rgb.length !== size * 3) {
    throw new RangeError(
      `expected ${size * 3} RGB bytes, got ${rgb.length}`
    );
  }
}

function checkGray(gray, size) {
  checkByteArray(gray, 'grayscale data');

  if (gray.length !== size) {
    throw new RangeError(
      `expected ${size} grayscale bytes, got ${gray.length}`
    );
  }
}

/**
 * PALIC/JPEG-LS MED:
 *
 *   C B
 *   A X
 *
 * row 0: left, with zero at x=0;
 * column 0: up.
 */
function predictMED(values, width, x, y) {
  if (x === 0 && y === 0) return 0;
  if (y === 0) return values[x - 1];
  if (x === 0) return values[(y - 1) * width];

  const i = y * width + x;
  const a = values[i - 1];
  const b = values[i - width];
  const c = values[i - width - 1];

  const min = Math.min(a, b);
  const max = Math.max(a, b);

  if (c >= max) return min;
  if (c <= min) return max;

  return a + b - c;
}

/**
 * Строковый вариант того же MED для кодера.
 * current/previous содержат уже восстановимые исходные значения компоненты.
 */
function predictRows(current, previous, x, y) {
  if (y === 0) {
    return x === 0 ? 0 : current[x - 1];
  }

  if (x === 0) {
    return previous[x];
  }

  const a = current[x - 1];
  const b = previous[x];
  const c = previous[x - 1];

  const min = Math.min(a, b);
  const max = Math.max(a, b);

  if (c >= max) return min;
  if (c <= min) return max;

  return a + b - c;
}

/**
 * history содержит последние width действительных z.
 * z текущего блока ещё не записаны.
 */
function chooseLForBlock(
  blockIndex,
  width,
  history,
  previousBlockSum,
  off
) {
  if (blockIndex === 0) return 2;

  const p = blockIndex * BLOCK_SIZE;
  let referenceSum;

  if (width < BLOCK_SIZE || p < width) {
    referenceSum = previousBlockSum;
  } else {
    referenceSum = 0;

    // p-width имеет тот же индекс в кольце, что и p.
    let cursor = p % width;

    for (let i = 0; i < BLOCK_SIZE; i++) {
      referenceSum += history[cursor];

      cursor++;
      if (cursor === width) cursor = 0;
    }
  }

  const average = Math.floor(referenceSum / BLOCK_SIZE);

  return Math.max(
    0,
    Math.min(7, bitWidth(average) - off)
  );
}

function createWriter(Type) {
  return {
    Type,
    data: new Type(0),
    length: 0
  };
}

function reserveWriter(writer, additional) {
  const required = writer.length + additional;
  let capacity = writer.data.length;

  if (required <= capacity) return;

  if (capacity === 0) capacity = 256;

  while (capacity < required) {
    const next = capacity * 2;

    if (!Number.isSafeInteger(next) || next <= capacity) {
      capacity = required;
      break;
    }

    capacity = next;
  }

  const data = new writer.Type(capacity);
  data.set(writer.data);
  writer.data = data;
}

function finishWriter(writer) {
  if (writer.data.length === writer.length) {
    return writer.data;
  }

  return writer.data.slice(0, writer.length);
}

function componentNames(format) {
  return format === FORMAT_PLN1
    ? PLN1_NAMES
    : PLN2_NAMES;
}

function createPlaneWork(
  format,
  planeIndex,
  width,
  size,
  blockCount,
  keepDiagnostics,
  keepZ
) {
  const luma = planeIndex === 0;
  const RowType = luma ? Uint8Array : Int16Array;
  const ZType = luma ? Uint8Array : Uint16Array;
  const DType = luma ? Int8Array : Int16Array;

  const topWriters = new Array(BUCKET_COUNT);
  const frequencyWork = new Array(BUCKET_COUNT);

  for (let L = 0; L < BUCKET_COUNT; L++) {
    // UV/L=0 временно Uint16Array. После обработки тайла тип
    // выбирается по maxSymbol, но не размер алфавита.
    const TopType =
      luma || L >= 1
        ? Uint8Array
        : Uint16Array;

    topWriters[L] = createWriter(TopType);

    const alphabetSize = luma
      ? 256 >>> L
      : 512 >>> L;

    // Всегда полный фиксированный алфавит.
    frequencyWork[L] = new Uint32Array(alphabetSize);
  }

  return {
    format,
    planeIndex,
    name: componentNames(format)[planeIndex],

    previousRow: new RowType(width),
    currentRow: new RowType(width),

    history: new ZType(width),
    historyPosition: 0,
    previousBlockSum: 0,

    blockZ: new ZType(BLOCK_SIZE),

    d0: keepDiagnostics ? new Int16Array(size) : null,
    d: keepDiagnostics ? new DType(size) : null,
    z: keepZ ? new ZType(size) : null,

    blockL: new Uint8Array(blockCount),
    rawOffsets: new Uint32Array(blockCount + 1),
    rawWriter: createWriter(Uint16Array),

    topWriters,
    frequencyWork,

    blocksByL: new Uint32Array(BUCKET_COUNT),
    maxSymbols: new Int32Array(BUCKET_COUNT).fill(-1),

    maxZ: 0,
    wrapped: 0
  };
}

function appendEncodedBlock(
  work,
  L,
  blockSum,
  blockIndex
) {
  const z = work.blockZ;
  const rawWriter = work.rawWriter;

  work.blockL[blockIndex] = L;
  work.rawOffsets[blockIndex] = rawWriter.length;

  reserveWriter(rawWriter, L);

  for (let bit = 0; bit < L; bit++) {
    let word = 0;

    for (let symbol = 0; symbol < BLOCK_SIZE; symbol++) {
      word |= ((z[symbol] >>> bit) & 1) << symbol;
    }

    rawWriter.data[rawWriter.length++] = word;
  }

  work.rawOffsets[blockIndex + 1] = rawWriter.length;

  const topWriter = work.topWriters[L];
  const frequencies = work.frequencyWork[L];

  reserveWriter(topWriter, BLOCK_SIZE);

  const output = topWriter.length;
  let maxSymbol = work.maxSymbols[L];

  for (let symbol = 0; symbol < BLOCK_SIZE; symbol++) {
    const top = z[symbol] >>> L;

    topWriter.data[output + symbol] = top;
    frequencies[top]++;

    if (top > maxSymbol) maxSymbol = top;
  }

  topWriter.length += BLOCK_SIZE;
  work.maxSymbols[L] = maxSymbol;
  work.blocksByL[L]++;
  work.previousBlockSum = blockSum;
}

function finishTopWriter(
  writer,
  planeIndex,
  L,
  maxSymbol
) {
  if (
    planeIndex !== 0 &&
    L === 0 &&
    maxSymbol <= 255
  ) {
    const result = new Uint8Array(writer.length);

    for (let i = 0; i < writer.length; i++) {
      result[i] = writer.data[i];
    }

    return result;
  }

  return finishWriter(writer);
}

function finishPlane(work) {
  const { planeIndex, format } = work;
  const luma = planeIndex === 0;
  const buckets = new Array(BUCKET_COUNT);

  for (let L = 0; L < BUCKET_COUNT; L++) {
    const writer = work.topWriters[L];
    const symbolCount = writer.length;
    const maxSymbol = work.maxSymbols[L];

    // PLN1/Y и PLN2/U/V имеют разные, но фиксированные алфавиты.
    // maxSymbol на alphabetSize не влияет.
    const alphabetSize = luma
      ? 256 >>> L
      : 512 >>> L;

    const coefficients = finishTopWriter(
      writer,
      planeIndex,
      L,
      maxSymbol
    );

    const bucket = {
      id: planeIndex * BUCKET_COUNT + L,
      plane: planeIndex,
      component: work.name,
      L,
      mode: 'fse',
      alphabetSize,
      maxSymbol,
      blockCount: work.blocksByL[L],
      symbolCount,
      coefficients,
      frequencies: work.frequencyWork[L]
    };

    buckets[L] = bucket;
  }

  return {
    name: work.name,
    residualBits: luma ? 8 : 9,

    d0: work.d0,
    d: work.d,
    z: work.z,

    blockCount: work.blockL.length,
    blockL: work.blockL,
    rawOffsets: work.rawOffsets,
    rawWords: finishWriter(work.rawWriter),
    buckets
  };
}

/**
 * Общая блочная механика, но не общий файловый формат:
 *
 * PLN1: один независимый grayscale-компонент, wrap8.
 * PLN2: RGB -> RCT -> Y/U/V, Y wrap8, U/V wrap9.
 */
function decomposeTile(
  input,
  width,
  height,
  options,
  format
) {
  const {
    size,
    blockCount,
    tailPadding
  } = getGeometry(width, height);

  if (format === FORMAT_PLN1) {
    checkGray(input, size);
  } else if (format === FORMAT_PLN2) {
    checkRGB(input, size);
  } else {
    throw new RangeError(`unsupported format: ${format}`);
  }

  const off = options.off ?? DEFAULT_OFF;
  const keepDiagnostics = options.keepDiagnostics === true;
  const keepZ = options.keepZ === true;

  validateOff(off);

  const planeCount = format === FORMAT_PLN1 ? 1 : 3;

  const work = Array.from(
    { length: planeCount },
    (_, planeIndex) =>
      createPlaneWork(
        format,
        planeIndex,
        width,
        size,
        blockCount,
        keepDiagnostics,
        keepZ
      )
  );

  const blockSums = new Array(planeCount);
  const blockLs = new Array(planeCount);

  let rasterX = 0;
  let rasterY = 0;
  let bufferedRow = 0;

  for (
    let blockIndex = 0;
    blockIndex < blockCount;
    blockIndex++
  ) {
    const blockStart = blockIndex * BLOCK_SIZE;

    for (let planeIndex = 0; planeIndex < planeCount; planeIndex++) {
      const plane = work[planeIndex];

      blockLs[planeIndex] = chooseLForBlock(
        blockIndex,
        width,
        plane.history,
        plane.previousBlockSum,
        off
      );

      blockSums[planeIndex] = 0;
    }

    for (let slot = 0; slot < BLOCK_SIZE; slot++) {
      const pixel = blockStart + slot;

      if (pixel >= size) {
        for (let planeIndex = 0; planeIndex < planeCount; planeIndex++) {
          work[planeIndex].blockZ[slot] = 0;
        }

        continue;
      }

      const x = rasterX;
      const y = rasterY;

      if (y !== bufferedRow) {
        for (const plane of work) {
          const previous = plane.previousRow;
          plane.previousRow = plane.currentRow;
          plane.currentRow = previous;
        }

        bufferedRow = y;
      }

      if (format === FORMAT_PLN1) {
        work[0].currentRow[x] = input[pixel];
      } else {
        const source = pixel * 3;
        const r = input[source];
        const g = input[source + 1];
        const b = input[source + 2];

        const u = r - g;
        const v0 = b - g;

        work[0].currentRow[x] =
          g + ((u + v0 + 2) >> 2);
        work[1].currentRow[x] = u;
        work[2].currentRow[x] =
          v0 - (u >> 2);
      }

      for (let planeIndex = 0; planeIndex < planeCount; planeIndex++) {
        const plane = work[planeIndex];
        const value = plane.currentRow[x];

        const prediction = predictRows(
          plane.currentRow,
          plane.previousRow,
          x,
          y
        );

        const originalDelta = value - prediction;
        const delta = planeIndex === 0
          ? wrap8(originalDelta)
          : wrap9(originalDelta);

        const encodedZ = zigzag(delta);

        plane.blockZ[slot] = encodedZ;
        blockSums[planeIndex] += encodedZ;

        plane.history[plane.historyPosition] = encodedZ;
        plane.historyPosition++;

        if (plane.historyPosition === width) {
          plane.historyPosition = 0;
        }

        if (encodedZ > plane.maxZ) {
          plane.maxZ = encodedZ;
        }

        if (
          planeIndex !== 0 &&
          delta !== originalDelta
        ) {
          plane.wrapped++;
        }

        if (keepDiagnostics) {
          plane.d0[pixel] = originalDelta;
          plane.d[pixel] = delta;
        }

        if (keepZ) {
          plane.z[pixel] = encodedZ;
        }
      }

      rasterX++;

      if (rasterX === width) {
        rasterX = 0;
        rasterY++;
      }
    }

    for (let planeIndex = 0; planeIndex < planeCount; planeIndex++) {
      appendEncodedBlock(
        work[planeIndex],
        blockLs[planeIndex],
        blockSums[planeIndex],
        blockIndex
      );
    }
  }

  const planes = work.map(finishPlane);

  let fseSymbolCount = 0;
  let rawBitPlaneWordCount = 0;

  for (const plane of planes) {
    rawBitPlaneWordCount += plane.rawWords.length;

    for (const bucket of plane.buckets) {
      fseSymbolCount += bucket.symbolCount;
    }
  }

  const result = {
    version: VERSION,
    format,
    scope: 'tile',
    width,
    height,
    off,
    blockSize: BLOCK_SIZE,
    blockLayout: BLOCK_LAYOUT,
    predictor: PREDICTOR,
    layout: 'planar-row-major',
    colorTransform:
      format === FORMAT_PLN2 ? COLOR_TRANSFORM : null,
    residualWrap:
      format === FORMAT_PLN2
        ? { Y: 8, U: 9, V: 9 }
        : { gray: 8 },
    rawBitOrder: 'symbol-i-to-word-bit-i',
    planes,

    stats: {
      pixels: size,
      blocksPerPlane: blockCount,
      totalBlocks: blockCount * planeCount,
      paddedSymbolsPerPlane: tailPadding,
      wrappedChroma:
        format === FORMAT_PLN2
          ? work[1].wrapped + work[2].wrapped
          : 0,
      maxZ: work.map(plane => plane.maxZ),
      Lhist: work.map(plane =>
        Array.from(plane.blocksByL)
      ),
      fseSymbolCount,
      rawBitPlaneWordCount
    }
  };

  return result;
}

/** Один PLN2 RGB-тайл. */
function decompose(rgb, width, height, options = {}) {
  return decomposeTile(
    rgb,
    width,
    height,
    options,
    FORMAT_PLN2
  );
}

/** Один PLN1 grayscale-тайл, без RCT. */
function decomposeGray(gray, width, height, options = {}) {
  return decomposeTile(
    gray,
    width,
    height,
    options,
    FORMAT_PLN1
  );
}

function requireArrayLike(value, label) {
  if (
    value == null ||
    typeof value.length !== 'number'
  ) {
    throw new TypeError(
      `${label} must be an array or typed array`
    );
  }

  return value;
}

function validateBucketForDecoding(
  bucket,
  planeIndex,
  L,
  name
) {
  if (
    !bucket ||
    bucket.plane !== planeIndex ||
    bucket.L !== L
  ) {
    throw new TypeError(
      `${name}: invalid bucket ${L}`
    );
  }

  requireArrayLike(
    bucket.coefficients,
    `${name}.buckets[${L}].coefficients`
  );

  if (
    bucket.coefficients.length %
    BLOCK_SIZE !== 0
  ) {
    throw new RangeError(
      `${name}: bucket ${L} has a partial coded block`
    );
  }

  if (
    bucket.coefficients.length !==
    bucket.symbolCount
  ) {
    throw new RangeError(
      `${name}: bucket ${L} length mismatch`
    );
  }

  const expectedAlphabet =
    planeIndex === 0
      ? 256 >>> L
      : 512 >>> L;

  if (bucket.alphabetSize !== expectedAlphabet) {
    throw new RangeError(
      `${name}: bucket ${L} has alphabet ` +
      `${bucket.alphabetSize}, expected ${expectedAlphabet}`
    );
  }

  if (bucket.mode !== 'fse') {
    throw new TypeError('invalid bucket mode');
  }

  if (
    !(bucket.frequencies instanceof Uint32Array) ||
    bucket.frequencies.length !== expectedAlphabet
  ) {
    throw new RangeError(
      `${name}: bucket ${L} must have a full ` +
      `${expectedAlphabet}-entry frequency array`
    );
  }
}

function decodePlane(
  encodedPlane,
  planeIndex,
  width,
  height,
  off,
  name,
  sharedRawReader = null
) {
  const {
    size,
    blockCount
  } = getGeometry(width, height);

  if (
    !encodedPlane ||
    !Array.isArray(encodedPlane.buckets) ||
    encodedPlane.buckets.length !== BUCKET_COUNT
  ) {
    throw new TypeError(
      `${name}: expected eight buckets`
    );
  }

  for (let L = 0; L < BUCKET_COUNT; L++) {
    validateBucketForDecoding(
      encodedPlane.buckets[L],
      planeIndex,
      L,
      name
    );
  }

  const rawWords = sharedRawReader
    ? null
    : requireArrayLike(
        encodedPlane.rawWords,
        `${name}.rawWords`
      );

  const rawBlockWords = sharedRawReader
    ? new Uint16Array(7)
    : null;

  const expectedL = encodedPlane.blockL;

  if (
    expectedL != null &&
    expectedL.length !== blockCount
  ) {
    throw new RangeError('blockL has invalid length');
  }

  const values = planeIndex === 0
    ? new Uint8Array(size)
    : new Int16Array(size);

  const ZType = planeIndex === 0
    ? Uint8Array
    : Uint16Array;

  const history = new ZType(width);
  const bucketOffsets =
    new Uint32Array(BUCKET_COUNT);
  const observedMax =
    new Int32Array(BUCKET_COUNT).fill(-1);

  const maxAllowedZ =
    planeIndex === 0 ? 255 : 511;

  let historyPosition = 0;
  let previousBlockSum = 0;
  let rawOffset = 0;
  let rasterX = 0;
  let rasterY = 0;

  for (
    let blockIndex = 0;
    blockIndex < blockCount;
    blockIndex++
  ) {
    const blockStart = blockIndex * BLOCK_SIZE;

    const L = chooseLForBlock(
      blockIndex,
      width,
      history,
      previousBlockSum,
      off
    );

    if (
      expectedL != null &&
      expectedL[blockIndex] !== L
    ) {
      throw new RangeError(
        `${name}: L mismatch at block ${blockIndex}: ` +
        `stored=${expectedL[blockIndex]}, computed=${L}`
      );
    }

    const bucket = encodedPlane.buckets[L];
    const coefficients = bucket.coefficients;
    const coefficientOffset = bucketOffsets[L];

    if (
      coefficientOffset + BLOCK_SIZE >
      coefficients.length
    ) {
      throw new RangeError(
        `${name}: bucket ${L} is truncated`
      );
    }

    if (sharedRawReader) {
      if (sharedRawReader.offset + L > sharedRawReader.words.length) {
        throw new RangeError(
          `${name}: shared RAW stream is truncated`
        );
      }

      for (let bit = 0; bit < L; bit++) {
        rawBlockWords[bit] =
          sharedRawReader.words[sharedRawReader.offset++];
      }
    } else if (rawOffset + L > rawWords.length) {
      throw new RangeError(
        `${name}: raw bit-plane words are truncated`
      );
    }

    let blockSum = 0;

    for (let slot = 0; slot < BLOCK_SIZE; slot++) {
      const top =
        coefficients[coefficientOffset + slot];

      if (
        !Number.isInteger(top) ||
        top < 0 ||
        top > 65535
      ) {
        throw new RangeError(
          'invalid top coefficient'
        );
      }

      if (top >= bucket.alphabetSize) {
        throw new RangeError(
          `symbol ${top} is outside alphabet ` +
          `${bucket.alphabetSize}`
        );
      }

      if (top > observedMax[L]) {
        observedMax[L] = top;
      }

      let lowBits = 0;

      for (let bit = 0; bit < L; bit++) {
        const word = sharedRawReader
          ? rawBlockWords[bit]
          : rawWords[rawOffset + bit];

        lowBits |= ((word >>> slot) & 1) << bit;
      }

      const encodedZ =
        (top << L) | lowBits;

      if (encodedZ > maxAllowedZ) {
        throw new RangeError(
          `${name}: invalid z=${encodedZ}`
        );
      }

      blockSum += encodedZ;

      const pixel = blockStart + slot;

      if (pixel >= size) {
        if (encodedZ !== 0) {
          throw new RangeError(
            `${name}: nonzero z in tail padding`
          );
        }

        continue;
      }

      history[historyPosition] = encodedZ;
      historyPosition++;

      if (historyPosition === width) {
        historyPosition = 0;
      }

      const x = rasterX;
      const y = rasterY;

      // Восстановленный полноразмерный MED декодера.
      const prediction = predictMED(
        values,
        width,
        x,
        y
      );

      const delta = unzigzag(encodedZ);

      const value = planeIndex === 0
        ? (prediction + delta) & 255
        : wrap9(prediction + delta);

      if (
        planeIndex !== 0 &&
        (value < -255 || value > 255)
      ) {
        throw new RangeError(
          `${name}: invalid restored value ${value}`
        );
      }

      values[pixel] = value;

      rasterX++;

      if (rasterX === width) {
        rasterX = 0;
        rasterY++;
      }
    }

    bucketOffsets[L] += BLOCK_SIZE;

    if (!sharedRawReader) {
      rawOffset += L;
    }

    previousBlockSum = blockSum;
  }

  for (let L = 0; L < BUCKET_COUNT; L++) {
    const bucket = encodedPlane.buckets[L];

    if (
      bucketOffsets[L] !==
      bucket.coefficients.length
    ) {
      throw new RangeError(
        `${name}: unused coefficients in bucket ${L}`
      );
    }

    if (observedMax[L] !== bucket.maxSymbol) {
      throw new RangeError(
        `${name}: maxSymbol mismatch in bucket ${L}`
      );
    }
  }

  if (!sharedRawReader && rawOffset !== rawWords.length) {
    throw new RangeError(
      `${name}: unused raw bit-plane words`
    );
  }

  return values;
}

function validateTileHeader(
  decomposition,
  expectedFormat
) {
  if (
    !decomposition ||
    decomposition.version !== VERSION
  ) {
    throw new RangeError(
      'unsupported decomposition version'
    );
  }

  if (decomposition.format !== expectedFormat) {
    throw new RangeError(
      `expected ${expectedFormat}, got ` +
      `${decomposition.format}`
    );
  }

  if (decomposition.scope !== 'tile') {
    throw new RangeError('expected a single tile');
  }

  if (
    decomposition.blockLayout !== BLOCK_LAYOUT
  ) {
    throw new RangeError(
      `incompatible block layout: ` +
      `${decomposition.blockLayout}`
    );
  }

  if (decomposition.blockSize !== BLOCK_SIZE) {
    throw new RangeError('unsupported block size');
  }

  if (decomposition.predictor !== PREDICTOR) {
    throw new RangeError('unsupported predictor');
  }

  if (expectedFormat === FORMAT_PLN2) {
    if (
      decomposition.colorTransform !==
      COLOR_TRANSFORM
    ) {
      throw new RangeError(
        'unsupported color transform'
      );
    }
  } else if (
    decomposition.colorTransform !== null
  ) {
    throw new RangeError(
      'PLN1 must not contain an RCT transform'
    );
  }
}

function decodeTileValues(
  decomposition,
  expectedFormat,
  sharedRawWords = null
) {
  validateTileHeader(
    decomposition,
    expectedFormat
  );

  const {
    width,
    height,
    off
  } = decomposition;

  validateOff(off);
  getGeometry(width, height);

  const expectedPlanes =
    expectedFormat === FORMAT_PLN1 ? 1 : 3;

  if (
    !Array.isArray(decomposition.planes) ||
    decomposition.planes.length !== expectedPlanes
  ) {
    throw new TypeError(
      `${expectedFormat}: expected ` +
      `${expectedPlanes} plane(s)`
    );
  }

  const names = componentNames(expectedFormat);
  const sharedRawReader = sharedRawWords == null
    ? null
    : {
        words: requireArrayLike(
          sharedRawWords,
          'shared RAW words'
        ),
        offset: 0
      };

  const values = names.map((name, planeIndex) =>
    decodePlane(
      decomposition.planes[planeIndex],
      planeIndex,
      width,
      height,
      off,
      name,
      sharedRawReader
    )
  );

  if (
    sharedRawReader &&
    sharedRawReader.offset !== sharedRawReader.words.length
  ) {
    throw new RangeError('unused words in shared RAW stream');
  }

  return values;
}

/** PLN2: восстановление RCT-плоскостей -> RGB. */
function recomposePLN2Values(decomposition, values) {
  const { width, height } = decomposition;
  const { size } = getGeometry(width, height);

  const Y = values[0];
  const U = values[1];
  const V = values[2];
  const rgb = new Uint8Array(size * 3);

  for (
    let i = 0, destination = 0;
    i < size;
    i++, destination += 3
  ) {
    const u = U[i];
    const v0 = V[i] + (u >> 2);
    const g = Y[i] - ((u + v0 + 2) >> 2);
    const r = u + g;
    const b = v0 + g;

    if (
      r < 0 || r > 255 ||
      g < 0 || g > 255 ||
      b < 0 || b > 255
    ) {
      throw new RangeError(
        `invalid restored RGB at pixel ${i}: ` +
        `${r},${g},${b}`
      );
    }

    rgb[destination] = r;
    rgb[destination + 1] = g;
    rgb[destination + 2] = b;
  }

  return rgb;
}

/** PLN2: один тайл -> RGB. */
function recompose(decomposition) {
  const values = decodeTileValues(
    decomposition,
    FORMAT_PLN2
  );

  return recomposePLN2Values(decomposition, values);
}

/** PLN2: один тайл -> RGB, один общий RAW-поток Y -> U -> V. */
function recomposeSharedRaw(decomposition, rawWords) {
  const values = decodeTileValues(
    decomposition,
    FORMAT_PLN2,
    rawWords
  );

  return recomposePLN2Values(decomposition, values);
}

/** PLN1: один тайл -> grayscale. */
function recomposeGray(decomposition) {
  const values = decodeTileValues(
    decomposition,
    FORMAT_PLN1
  );

  return values[0];
}

/** PLN1: один тайл -> grayscale, один общий RAW-поток. */
function recomposeGraySharedRaw(decomposition, rawWords) {
  const values = decodeTileValues(
    decomposition,
    FORMAT_PLN1,
    rawWords
  );

  return values[0];
}

function extractTile(
  input,
  imageWidth,
  x,
  y,
  tileWidth,
  tileHeight,
  channels
) {
  const output = new Uint8Array(
    tileWidth * tileHeight * channels
  );

  const rowBytes = tileWidth * channels;

  for (let row = 0; row < tileHeight; row++) {
    const source =
      ((y + row) * imageWidth + x) * channels;

    output.set(
      input.subarray(
        source,
        source + rowBytes
      ),
      row * rowBytes
    );
  }

  return output;
}

/**
 * Полное изображение разбивается на row-major независимые тайлы.
 * Каждый вызов decompose/decomposeGray создаёт новые MED-строки,
 * history, previousBlockSum и blockL; состояние между тайлами
 * не разделяется.
 */
function decomposeTiled(
  input,
  width,
  height,
  options,
  format
) {
  const { size } = getGeometry(width, height);
  const channels = format === FORMAT_PLN1 ? 1 : 3;

  if (format === FORMAT_PLN1) {
    checkGray(input, size);
  } else {
    checkRGB(input, size);
  }

  const off = options.off ?? DEFAULT_OFF;
  validateOff(off);

  const tiles = [];

  for (let y = 0; y < height; y += TILE_SIZE) {
    const tileHeight =
      Math.min(TILE_SIZE, height - y);

    for (let x = 0; x < width; x += TILE_SIZE) {
      const tileWidth =
        Math.min(TILE_SIZE, width - x);

      const tileInput = extractTile(
        input,
        width,
        x,
        y,
        tileWidth,
        tileHeight,
        channels
      );

      const data = format === FORMAT_PLN1
        ? decomposeGray(
            tileInput,
            tileWidth,
            tileHeight,
            options
          )
        : decompose(
            tileInput,
            tileWidth,
            tileHeight,
            options
          );

      tiles.push({
        x,
        y,
        width: tileWidth,
        height: tileHeight,
        data
      });
    }
  }

  return {
    version: VERSION,
    format,
    scope: 'image',
    width,
    height,
    off,
    tileSize: TILE_SIZE,
    tileOrder: 'row-major-independent',
    tiles
  };
}

function decomposeImage(
  rgb,
  width,
  height,
  options = {}
) {
  return decomposeTiled(
    rgb,
    width,
    height,
    options,
    FORMAT_PLN2
  );
}

function decomposeGrayImage(
  gray,
  width,
  height,
  options = {}
) {
  return decomposeTiled(
    gray,
    width,
    height,
    options,
    FORMAT_PLN1
  );
}

function recomposeTiled(
  image,
  expectedFormat
) {
  if (
    !image ||
    image.version !== VERSION ||
    image.scope !== 'image'
  ) {
    throw new RangeError(
      'unsupported tiled image header'
    );
  }

  if (image.format !== expectedFormat) {
    throw new RangeError(
      `expected ${expectedFormat}, got ${image.format}`
    );
  }

  if (image.tileSize !== TILE_SIZE) {
    throw new RangeError('unsupported tile size');
  }

  if (image.tileOrder !== 'row-major-independent') {
    throw new RangeError('unsupported tile order');
  }

  const { width, height, off } = image;
  validateOff(off);

  const { size } = getGeometry(width, height);
  const channels =
    expectedFormat === FORMAT_PLN1 ? 1 : 3;

  if (!Array.isArray(image.tiles)) {
    throw new TypeError('tiles must be an array');
  }

  const output = new Uint8Array(size * channels);
  let tileIndex = 0;

  for (let y = 0; y < height; y += TILE_SIZE) {
    const tileHeight =
      Math.min(TILE_SIZE, height - y);

    for (let x = 0; x < width; x += TILE_SIZE) {
      const tileWidth =
        Math.min(TILE_SIZE, width - x);

      const entry = image.tiles[tileIndex++];

      if (
        !entry ||
        entry.x !== x ||
        entry.y !== y ||
        entry.width !== tileWidth ||
        entry.height !== tileHeight ||
        !entry.data ||
        entry.data.width !== tileWidth ||
        entry.data.height !== tileHeight ||
        entry.data.off !== off
      ) {
        throw new RangeError(
          `invalid tile entry at x=${x}, y=${y}`
        );
      }

      // Каждый вызов начинает декодирование с нового MED/history/L.
      const restored =
        expectedFormat === FORMAT_PLN1
          ? recomposeGray(entry.data)
          : recompose(entry.data);

      const rowBytes = tileWidth * channels;

      for (let row = 0; row < tileHeight; row++) {
        const destination =
          ((y + row) * width + x) * channels;

        output.set(
          restored.subarray(
            row * rowBytes,
            (row + 1) * rowBytes
          ),
          destination
        );
      }
    }
  }

  if (tileIndex !== image.tiles.length) {
    throw new RangeError('unused tile entries');
  }

  return output;
}

function recomposeImage(image) {
  return recomposeTiled(image, FORMAT_PLN2);
}

function recomposeGrayImage(image) {
  return recomposeTiled(image, FORMAT_PLN1);
}

function storageStats(decomposition) {
  const seen = new Set();

  const result = {
    diagnosticsBytes: 0,
    zBytes: 0,
    blockMetadataBytes: 0,
    rawWordsBytes: 0,
    coefficientBytes: 0,
    frequencyBytes: 0,
    totalTypedArrayBytes: 0,
    bytesPerPixel: 0
  };

  function add(array, field) {
    if (array == null || seen.has(array)) return;

    if (
      !ArrayBuffer.isView(array) ||
      array instanceof DataView
    ) {
      throw new TypeError(
        'storageStats expects TypedArray fields'
      );
    }

    seen.add(array);
    result[field] += array.byteLength;
    result.totalTypedArrayBytes += array.byteLength;
  }

  for (const plane of decomposition.planes) {
    add(plane.d0, 'diagnosticsBytes');
    add(plane.d, 'diagnosticsBytes');
    add(plane.z, 'zBytes');
    add(plane.blockL, 'blockMetadataBytes');
    add(plane.rawOffsets, 'blockMetadataBytes');
    add(plane.rawWords, 'rawWordsBytes');

    for (const bucket of plane.buckets) {
      add(
        bucket.coefficients,
        'coefficientBytes'
      );
      add(
        bucket.frequencies,
        'frequencyBytes'
      );
    }
  }

  result.bytesPerPixel =
    result.totalTypedArrayBytes /
    (decomposition.width * decomposition.height);

  return result;
}

module.exports = {
  // PLN2, один RGB-тайл.
  decompose,
  recompose,
  decomposePLN2: decompose,
  recomposePLN2: recompose,

  // PLN1, один grayscale-тайл.
  decomposeGray,
  recomposeGray,
  recomposeSharedRaw,
  recomposeGraySharedRaw,
  decomposePLN1: decomposeGray,
  recomposePLN1: recomposeGray,

  // Полные изображения с независимыми тайлами 288.
  decomposeImage,
  recomposeImage,
  decomposeGrayImage,
  recomposeGrayImage,

  storageStats,

  rct,
  irct,
  wrap8,
  wrap9,
  zigzag,
  unzigzag,
  bitWidth,

  constants: {
    VERSION,
    BLOCK_SIZE,
    BUCKET_COUNT,
    DEFAULT_OFF,
    TILE_SIZE,
    FORMAT_PLN1,
    FORMAT_PLN2,
    COLOR_TRANSFORM,
    PREDICTOR,
    BLOCK_LAYOUT
  }
};
