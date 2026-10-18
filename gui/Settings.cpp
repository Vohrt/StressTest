#include "pch.h"
#include "Settings.h"

#include <fstream>
#include <string>

namespace gui {
namespace {

std::string trim(std::string s) {
    while (!s.empty() && (s.back() == '\r' || s.back() == '\n' || s.back() == ' ' || s.back() == '\t')) {
        s.pop_back();
    }
    size_t start = 0;
    while (start < s.size() && (s[start] == ' ' || s[start] == '\t')) ++start;
    return s.substr(start);
}

double to_double(std::string const& s, double fallback) {
    try {
        return std::stod(s);
    } catch (...) {
        return fallback;
    }
}

int to_int(std::string const& s, int fallback) {
    try {
        return std::stoi(s);
    } catch (...) {
        return fallback;
    }
}

bool to_bool(std::string const& s, bool fallback) {
    if (s == "1" || s == "true") return true;
    if (s == "0" || s == "false") return false;
    return fallback;
}

std::vector<std::string> split_list(std::string const& value) {
    std::vector<std::string> out;
    size_t pos = 0;
    while (pos <= value.size()) {
        const size_t comma = value.find(',', pos);
        const std::string id =
            value.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        if (!id.empty()) out.push_back(id);
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return out;
}

} // namespace

std::wstring settings_path() {
    wchar_t buffer[MAX_PATH] = {};
    const DWORD n = GetEnvironmentVariableW(L"LOCALAPPDATA", buffer, MAX_PATH);
    std::wstring dir = (n > 0 && n < MAX_PATH) ? std::wstring(buffer, n) : L".";
    dir += L"\\CpuStressTestGui";
    CreateDirectoryW(dir.c_str(), nullptr);
    return dir + L"\\settings.ini";
}

Settings load_settings() {
    Settings s;
    std::ifstream in(settings_path());
    if (!in) return s;

    std::string line;
    while (std::getline(in, line)) {
        const size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = trim(line.substr(0, eq));
        const std::string value = trim(line.substr(eq + 1));
        if (key == "seconds") s.seconds = to_double(value, s.seconds);
        else if (key == "minutes") s.minutes = to_double(value, s.minutes);
        else if (key == "threads") s.threads = to_int(value, s.threads);
        else if (key == "sustained") s.sustained = to_bool(value, s.sustained);
        else if (key == "accept_reference") s.accept_reference = to_bool(value, s.accept_reference);
        else if (key == "no_verify") s.no_verify = to_bool(value, s.no_verify);
        else if (key == "csv") s.csv = to_bool(value, s.csv);
        else if (key == "csv_path") s.csv_path = value;
        else if (key == "all_selected") s.all_selected = to_bool(value, s.all_selected);
        else if (key == "selected") s.selected = split_list(value);
        else if (key == "window_width") s.window_width = to_int(value, s.window_width);
        else if (key == "window_height") s.window_height = to_int(value, s.window_height);
    }
    return s;
}

void save_settings(Settings const& s) {
    std::ofstream out(settings_path(), std::ios::trunc);
    if (!out) return;
    out << "seconds=" << s.seconds << "\n";
    out << "minutes=" << s.minutes << "\n";
    out << "threads=" << s.threads << "\n";
    out << "sustained=" << (s.sustained ? 1 : 0) << "\n";
    out << "accept_reference=" << (s.accept_reference ? 1 : 0) << "\n";
    out << "no_verify=" << (s.no_verify ? 1 : 0) << "\n";
    out << "csv=" << (s.csv ? 1 : 0) << "\n";
    out << "csv_path=" << s.csv_path << "\n";
    out << "window_width=" << s.window_width << "\n";
    out << "window_height=" << s.window_height << "\n";
    out << "all_selected=" << (s.all_selected ? 1 : 0) << "\n";
    out << "selected=";
    for (size_t i = 0; i < s.selected.size(); ++i) {
        if (i) out << ',';
        out << s.selected[i];
    }
    out << "\n";
}

} // namespace gui
