// Independent supplied-rule enumeration: ordinary queen bitboards plus direct camel tests.
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
class QueensCamel {
    int q;
    uint32_t full;
    std::vector<uint8_t> columns;
    std::vector<std::vector<uint8_t>> answers;
    void visit(int row, uint32_t used, uint32_t left, uint32_t right) {
        if (row == q) { answers.push_back(columns); return; }
        uint32_t choices = full & ~(used | left | right);
        while (choices) {
            const uint32_t bit = choices & -choices;
            choices ^= bit;
            const int col = __builtin_ctz(bit);
            if (row >= 1 && std::abs(col - columns[row-1]) == 3) continue;
            if (row >= 3 && std::abs(col - columns[row-3]) == 1) continue;
            columns[row] = static_cast<uint8_t>(col);
            visit(row+1, used|bit, (left|bit)<<1, (right|bit)>>1);
        }
    }
public:
    explicit QueensCamel(int n) : q(n), full((1U<<n)-1U), columns(n) {}
    void run(const char* path) {
        visit(0,0,0,0);
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        for (const auto& row : answers) out.write(reinterpret_cast<const char*>(row.data()), q);
        if (!out) throw std::runtime_error("Write failed");
        std::cout << answers.size() << '\n';
    }
};
int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::runtime_error("Usage: truth_camel q output");
        const int q = std::atoi(argv[1]);
        if (q < 1 || q > 20) throw std::runtime_error("q out of range");
        QueensCamel(q).run(argv[2]);return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n';return 1; }
}
