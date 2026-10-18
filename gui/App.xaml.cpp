#include "pch.h"
#include "App.xaml.h"
#include "MainWindow.xaml.h"

#include "cli.h"   // cst::cli_main

#include <string>
#include <vector>

using namespace winrt;
using namespace Microsoft::UI::Xaml;

namespace winrt::CpuStressTestGui::implementation
{
    App::App()
    {
        InitializeComponent();
    }

    void App::OnLaunched(LaunchActivatedEventArgs const&)
    {
        window = make<MainWindow>();
        window.Activate();
    }
}

// Provided by the XAML-generated App.xaml.g.hpp, which is compiled with
// DISABLE_XAML_GENERATED_MAIN so this file can supply the entry point.  The
// generated helper still does the real work (init_apartment + Application::Start).
int __stdcall wXamlGeneratedMain();

namespace {

// This is a GUI-subsystem executable, so the command line mode has to attach to
// the console (or keep the inherited pipes) before the C runtime can print.
// Without this, output would be lost when the tool is started from a console;
// real inherited handles (cmd redirection, a PowerShell pipe) are left alone.
//
// Note that PowerShell's own ">" operator cannot be supported from this side:
// PowerShell 5.1 does not read the output of a GUI-subsystem process, so the
// command returns at once, the pipe is closed and the file stays empty no
// matter what happens here.  The README recommends cmd redirection, a pipe or
// --csv instead.
void attach_console_streams()
{
    const HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
    if (out && out != INVALID_HANDLE_VALUE) return;   // pipes/files already work

    if (!AttachConsole(ATTACH_PARENT_PROCESS)) return;
    FILE* stream = nullptr;
    freopen_s(&stream, "CONOUT$", "w", stdout);
    freopen_s(&stream, "CONOUT$", "w", stderr);
    freopen_s(&stream, "CONIN$", "r", stdin);
}

std::string to_utf8(wchar_t const* wide)
{
    if (!wide || !*wide) return {};
    const int size = WideCharToMultiByte(CP_UTF8, 0, wide, -1, nullptr, 0, nullptr, nullptr);
    if (size <= 0) return {};
    std::string out(static_cast<size_t>(size - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide, -1, out.data(), size, nullptr, nullptr);
    return out;
}

// The GUI runs its measurements from the repository (that is where
// reference.txt, calibration.txt and results live), no matter where the exe was
// started from.  The command line keeps the caller's working directory.
void use_repository_directory()
{
    wchar_t exe[MAX_PATH] = {};
    const DWORD length = GetModuleFileNameW(nullptr, exe, MAX_PATH);
    if (length == 0 || length >= MAX_PATH) return;

    std::wstring directory(exe, length);
    const size_t slash = directory.find_last_of(L"\\/");
    if (slash == std::wstring::npos) return;
    directory.resize(slash);

    // build\ -> repository root (the folder that holds reference.txt).
    std::wstring probe = directory;
    for (int level = 0; level < 4; ++level) {
        if (GetFileAttributesW((probe + L"\\reference.txt").c_str()) != INVALID_FILE_ATTRIBUTES) {
            SetCurrentDirectoryW(probe.c_str());
            return;
        }
        const size_t parent = probe.find_last_of(L"\\/");
        if (parent == std::wstring::npos || parent < 3) return;
        probe.resize(parent);
    }
}

bool wants_gui(int argc, wchar_t** argv)
{
    if (argc <= 1) return true;
    const std::wstring first = argv[1];
    return first == L"gui" || first == L"--gui";
}

} // namespace

int __stdcall wWinMain(HINSTANCE, HINSTANCE, PWSTR, int)
{
    int argc = 0;
    LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);

    if (wants_gui(argc, argv)) {
        if (argv) LocalFree(argv);
        use_repository_directory();
        return wXamlGeneratedMain();
    }

    attach_console_streams();

    // 1:1 rebuild of argv in UTF-8, so the command line front end sees exactly
    // what the shell passed.
    std::vector<std::string> arguments;
    arguments.reserve(static_cast<size_t>(argc));
    arguments.push_back("cst");
    for (int i = 1; i < argc; ++i) arguments.push_back(to_utf8(argv[i]));
    if (argv) LocalFree(argv);

    std::vector<char*> raw;
    raw.reserve(arguments.size());
    for (std::string& argument : arguments) raw.push_back(argument.data());
    return cst::cli_main(static_cast<int>(raw.size()), raw.data());
}
