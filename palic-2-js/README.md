# palic-2-js

Текстовый вид корзин [palic-2](../palic-2) и программы, которые жмут те же корзины. Раскладка та же: тайл 288, `off = 2`, `L` считается по символам сверху и не пишется, RCT, MED, `wrap8` у Y и `wrap9` у U/V, зигзаг, блоки по 16 символов через строку. У плоскости восемь корзин `L0..L7` и маски. Символ 511 при `L = 0` у хромы — обычный символ алфавита 512.

Node.js 18+, зависимостей нет. Текст для глаз — [palicjs-text.js](palicjs-text.js). Сжатие — [palicjs-cli.js](palicjs-cli.js).

```bash
node palicjs-text.js image.ppm
```

`palicjs-text` печатает `PLN-TXT` картинки и ничего не жмёт.

Из кода: `renderTxt` пишет `PLN2-TXT`, `renderGrayTxt` — `PLN1-TXT`. `parseTxt` читает их обратно. Энтропия сюда не входит.

## PLN2-TXT

```text
PLN2-TXT
W H
4 13 22
L0 -
L1 11 14
M 20
```

Строка 1 — формат и метод. P6 это `PLN2`, P5 это `PLN1`. Строка 2 — размер картинки. Дальше индекс: номера строк, где в этом файле начинаются Y, U и V, через пробел. У серого `PLN1-TXT` в индексе один номер, не три. Тайлы одного ряда картинки стоят на одной строке индекса и разделены табом. Размер тайла из сетки, в файл не пишется:

```text
nx = (W + 287) / 288
ny = (H + 287) / 288
```

Потом плоскости по порядку Y, U, V. У каждой восемь строк `L0`…`L7` и строка `M`. В `PLN-TXT` там прогоны (`11` — один символ 1, `F` — выход в varint), не сырые десятичные числа. Пусто — `-`.

Кодек — библиотека [tile-fse-layout.js](tile-fse-layout.js). Программы отличаются только последним шагом, отладочным текстом и строкой метода в заголовке.

```bash
node palicjs-text.js image.ppm
node palicjs-cli.js -e rlgr image.ppm -o image.pjs
node palicjs-cli.js -d image.pjs -o image.ppm
```

`palicjs-text` пишет `PLN-TXT` и ничего не жмёт. `-e имя` загружает `entropy_имя.js`. Новый метод — это новый файл с `encode(Uint8Array|Uint16Array)` и `decode(bytes, length, bits)`, без правки программы.

Заголовок и есть формат с методом: `PLN2-RLGR`, `PLN1-TXT`, `PLN2-NONE`, `PLN1-RL`, `PLN2-RICE`. Вторая строка — `ширина высота`, дальше байты корзин или текст. Число плоскостей из `PLN1` или `PLN2`.

| Модуль | Метод |
|---|---|
| [entropy_none.js](entropy_none.js) | байты как есть. `bzip2 -9` этого файла — эталон |
| [entropy_rlgr.js](entropy_rlgr.js) | RLGR из llrice |
| [entropy_rl.js](entropy_rl.js) | ниблы и прогон нулей |
| [entropy_rice.js](entropy_rice.js) | Райс, `k` по среднему трёх |

## Свой модуль

Имя из `-e` — это имя файла. `node palicjs-cli.js -e foo image.ppm` загружает [entropy_foo.js](entropy_foo.js) и пишет заголовок `PLN2-FOO` или `PLN1-FOO`. Программу править не нужно. Имя: строчные латинские буквы и цифры, с буквы.

Модуль жмёт один массив и ничего не знает про картинку. Длина и ширина символа лежат в контейнере, не в твоих байтах.

```js
'use strict';

function encode(values) {
  if (!(values instanceof Uint8Array) && !(values instanceof Uint16Array)) {
    throw new TypeError('expected Uint8Array or Uint16Array');
  }
  return new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
}

function decode(bytes, length, bits) {
  if (bits === 8) return bytes.slice(0, length);
  const out = new Uint16Array(length);
  for (let i = 0; i < length; i++) out[i] = bytes[i * 2] | (bytes[i * 2 + 1] << 8);
  return out;
}

module.exports = { encode, decode };
```

`encode` получает `Uint8Array` или `Uint16Array` и возвращает `Uint8Array`. `decode(bytes, length, bits)` возвращает столько же символов: `bits` равен 8 или 16. Пустой массив законен, для него `encode` возвращает пустые байты.

Проверка, прежде чем мерить файл:

```js
const back = decode(encode(values), values.length, values instanceof Uint16Array ? 16 : 8);
```

`back` должен совпасть с `values` побайтно. Потом:

```bash
node palicjs-cli.js -e foo kodim23.ppm -o kodim23.pjs
node palicjs-cli.js -d kodim23.pjs -o kodim23.back.ppm
```

Пиксели `kodim23.back.ppm` и исходного P6 должны совпасть. В таблицу ниже попадает размер `.pjs`, серый считается на P5 и в рейтинг места не идёт.

## Рейтинг, kodim23

`kodak_color_p6_full.zip` и `kodak_gray_all.zip`, кадр 500×500. Пиксели после `-d` совпали.

| Место | Метод | Цвет | Серый | Цвет, бит/пикс | Серый, бит/пикс |
|---|---|---:|---:|---:|---:|
| 1 | palic-2 | 293 188 | 121 460 | 9.38 | 3.89 |
| 2 | `-e rlgr` | 307 350 | 125 162 | 9.84 | 4.01 |
| 3 | bz2 -9 файла `-e none` | 326 616 | 133 996 | 10.45 | 4.29 |
| 4 | `-e rice` | 336 043 | 141 611 | 10.75 | 4.53 |
| 5 | `-e rl` | 415 721 | 164 098 | 13.30 | 5.25 |
| 6 | P6 / P5 | 750 015 | 250 015 | 24.00 | 8.00 |
| 7 | `-e none` | 781 218 | 274 593 | 25.00 | 8.79 |
