// Independent reverse-row literal-F enumeration. No forward-row masks.
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
class Enumerator {
    int q, n;
    std::vector<uint8_t> F, columns;
    std::vector<int> selected;
    std::vector<std::vector<uint8_t>> answers;
    void visit(int row) {
        if (row < 0) { answers.push_back(columns); return; }
        for (int col = 0; col < q; ++col) {
            const int cell = q * row + col;
            bool conflict = false;
            for (const int old : selected) {
                if (F[cell * n + old]) { conflict = true; break; }
            }
            if (conflict) continue;
            columns[row] = static_cast<uint8_t>(col);
            selected.push_back(cell);
            visit(row - 1);
            selected.pop_back();
        }
    }
public:
    explicit Enumerator(int q_) : q(q_), n(q*q), F(n*n), columns(q) {}
    void run(const std::string& input, const std::string& output) {
        std::ifstream in(input, std::ios::binary);
        if (!in.read(reinterpret_cast<char*>(F.data()), F.size())) throw std::runtime_error("Truncated F");
        char extra;
        if (in.read(&extra, 1)) throw std::runtime_error("Oversized F");
        for (auto x : F) if (x > 1) throw std::runtime_error("Nonbinary F");
        visit(q-1);
        std::sort(answers.begin(), answers.end());
        std::ofstream out(output, std::ios::binary | std::ios::trunc);
        for (const auto& row : answers) out.write(reinterpret_cast<const char*>(row.data()), q);
        if (!out) throw std::runtime_error("Output write failed");
        std::cout << answers.size() << '\n';
    }
};
int main(int argc, char** argv) {
    try {
        if (argc != 4) throw std::runtime_error("Usage: reverse q F.bin columns.bin");
        int q = std::stoi(argv[1]);
        if (q < 1 || q > 255) throw std::runtime_error("q out of uint8 range");
        Enumerator(q).run(argv[2], argv[3]);
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
