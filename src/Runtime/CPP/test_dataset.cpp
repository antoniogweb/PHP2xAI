#include <iostream>
#include <vector>
#include <string>

#include "Dataset/stream_file_dataset.hpp"

int main(int argc, char** argv) {
    try {
        // uso: ./test_ds train.txt 32 [--sample]
        std::string path = (argc >= 2) ? argv[1] : "train.txt";
        std::size_t batchSize = (argc >= 3) ? static_cast<std::size_t>(std::stoul(argv[2])) : 32;
        bool sampleMode = false;
        for (int i = 3; i < argc; ++i) {
            if (std::string(argv[i]) == "--sample") {
                sampleMode = true;
            }
        }

        PHP2xAI::Runtime::CPP::StreamFileDataset ds(path, batchSize, '|', 42);

        // Simulo 1 epoca (puoi mettere un for(epoch) se vuoi)
        ds.shuffleEpoch();

        std::size_t batchIdx = 0;
        while (ds.nextBatch()) {
            std::cout << "\n=== BATCH " << batchIdx << " ===\n";

            if (sampleMode) {
                std::vector<float> x, y;
                std::size_t sampleIdx = 0;

                while (ds.nextSampleInBatch(x, y)) {
                    std::cout << "sample " << sampleIdx << " ";
                    PHP2xAI::Runtime::CPP::StreamFileDataset::printVec("X", x);
                    std::cout << " ";
                    PHP2xAI::Runtime::CPP::StreamFileDataset::printVec("Y", y);
                    std::cout << "\n";
                    ++sampleIdx;
                }
            } else {
                std::vector<float> xPacked, yPacked;
                ds.pack(xPacked, yPacked);

                PHP2xAI::Runtime::CPP::StreamFileDataset::printVec("X", xPacked);
                std::cout << " ";
                PHP2xAI::Runtime::CPP::StreamFileDataset::printVec("Y", yPacked);
                std::cout << "\n";
            }

            std::cout << "=== END BATCH " << batchIdx << " ===\n";
            ++batchIdx;
        }

        std::cout << "\nDone. Total batches iterated: " << batchIdx << "\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
