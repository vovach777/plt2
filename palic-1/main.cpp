#include <iostream>
#include <string>
#include <cassert>
#include <fstream>
#include <vector>
#include <array>
#include "netpbm.hpp"
#include "profiling.hpp"
#include "rgb.hpp"

int main(int argc, char** argv) {

    if (argc < 3) {
        std::cout << "PALIC Converter - RGB24 to plannar Rx16 Gx16 Bx16" << std::endl;
        std::cout << "> github.com/vovach777" << std::endl;
        std::cout << "Only .ppm (P6) types are supported for input!" << std::endl;
        std::cout << "Please use this format: \"" << "p-conv" << " input_file.pnm output_file.rgb\"" << std::endl;
        return 0;
    }
    try {
        const char* filename = argv[1];
        const char* outputFilename = argv[2];
        auto p6 = Netpbm(filename);

        if (p6.type != '6') {
            throw std::runtime_error("P6 only support");
        }
        if (p6.maxvalue != 255) {
            throw std::runtime_error("rgb24 only support");
        }
        if (p6.width < 16 ) {   // FIX: one 48-byte block = 16 px is enough
            throw std::runtime_error("uncompresable");
        }

        std::ofstream out(outputFilename, std::ios::binary);
        out.exceptions(std::ios::failbit | std::ios::badbit);

        profiling::StopWatch sw;
        auto line = std::vector<uint8_t>( p6.width * 3 );
        size_t last_block_size = p6.width*3 % 48;
        size_t blocks_per_line = p6.width*3 / 48 + (last_block_size != 0);
        size_t component_size = blocks_per_line * 16;
        auto line_planar = std::vector(3, std::vector( component_size ,int16_t{}));


        for (size_t h = 0; h < p6.height; ++h) {
            p6.read(line.data(), line.size());
            sw.start();
            const uint8_t* line_u8 = line.data();
            for (size_t n=0; n < blocks_per_line; n++) {
                auto offset = (n == blocks_per_line - 1 && last_block_size != 0) * (48-last_block_size);
                conv::convert_to_plannar_line_8to16_48(line.data() +n*48 -offset, &line_planar[0][n*16 - offset/3],&line_planar[1][n*16 - offset/3], &line_planar[2][n*16 - offset/3]);   // FIX: the tail block overlaps in the planes too
            }
            sw.stop();
            out.write(reinterpret_cast<const char*>(line_planar[0].data()),component_size*2);
            out.write(reinterpret_cast<const char*>(line_planar[1].data()),component_size*2);
            out.write(reinterpret_cast<const char*>(line_planar[2].data()),component_size*2);

        }

        out.close();
        sw.stop();
        auto elapsed = sw.elapsed();
        std::cout << "Conversion completed in " << elapsed << " seconds." << std::endl;
        std::cout << "Converted " << p6.width << "x" << p6.height << " image." << std::endl;
        if (last_block_size != 48) {
            std::cout << "Last block overlap: " << last_block_size << std::endl;
        }
        std::cout << (p6.width * p6.height * 3) / (elapsed * 1024 * 1024) <<  " MiB/sec." << std::endl;


    } catch (const std::exception& e) {
        std::cerr << "Error compressing image: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return 2;
    }
    return 0;
}
