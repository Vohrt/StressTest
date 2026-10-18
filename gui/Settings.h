// GUI settings, persisted to %LOCALAPPDATA%\CpuStressTestGui\settings.ini.
//
// Deliberately a tiny key=value format: no parser dependency, and it stays
// readable if someone wants to hand-edit it.
#pragma once

#include <string>
#include <vector>

namespace gui {

struct Settings {
    double seconds = 15.0;
    double minutes = 20.0;
    int threads = 0;
    bool sustained = false;
    // Note: "record reference" and "skip verification" are deliberately NOT
    // stored here.  Persisting either one would silently disable verification
    // of every later session; the record action is a one-shot button with a
    // confirmation dialog, and the skip-verification box starts unchecked on
    // every launch.
    bool csv = false;
    std::string csv_path = "results.csv";
    // Selection: all_selected covers the default "everything checked" state,
    // selected carries the explicit list otherwise (possibly empty = none).
    bool all_selected = true;
    std::vector<std::string> selected;
    int window_width = 1240;
    int window_height = 860;
};

std::wstring settings_path();
Settings load_settings();
void save_settings(Settings const& s);

} // namespace gui
