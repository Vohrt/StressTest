#pragma once

// MainWindow is declared in Project.idl, so the C++/WinRT generated header
// comes first; it pulls in the XAML-generated MainWindow.xaml.g.h itself.
#include "MainWindow.g.h"

#include "Settings.h"
#include "session.h"   // cst::TestOutcome, used by the session callbacks

#include <winrt/Microsoft.UI.Xaml.Controls.h>

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace winrt::CpuStressTestGui::implementation
{
    // One row of the test table: the widgets that show this workload's result.
    struct TestRow
    {
        std::string id;
        winrt::Microsoft::UI::Xaml::Controls::CheckBox check{ nullptr };
        winrt::Microsoft::UI::Xaml::Controls::TextBlock value{ nullptr };
        winrt::Microsoft::UI::Xaml::Controls::TextBlock index{ nullptr };
        winrt::Microsoft::UI::Xaml::Controls::TextBlock status{ nullptr };
    };

    struct MainWindow : MainWindowT<MainWindow>
    {
        MainWindow();

        // XAML event handlers.
        void OnRunSelected(winrt::Windows::Foundation::IInspectable const&,
                           winrt::Microsoft::UI::Xaml::RoutedEventArgs const&);
        void OnRunAll(winrt::Windows::Foundation::IInspectable const&,
                      winrt::Microsoft::UI::Xaml::RoutedEventArgs const&);
        void OnStop(winrt::Windows::Foundation::IInspectable const&,
                    winrt::Microsoft::UI::Xaml::RoutedEventArgs const&);
        void OnSelectAll(winrt::Windows::Foundation::IInspectable const&,
                         winrt::Microsoft::UI::Xaml::RoutedEventArgs const&);
        void OnSelectNone(winrt::Windows::Foundation::IInspectable const&,
                          winrt::Microsoft::UI::Xaml::RoutedEventArgs const&);
        void OnModeChanged(winrt::Windows::Foundation::IInspectable const&,
                           winrt::Microsoft::UI::Xaml::RoutedEventArgs const&);
        void OnCsvToggled(winrt::Windows::Foundation::IInspectable const&,
                          winrt::Microsoft::UI::Xaml::RoutedEventArgs const&);
        void OnClosed(winrt::Windows::Foundation::IInspectable const&,
                      winrt::Microsoft::UI::Xaml::WindowEventArgs const&);

    private:
        void build_test_rows();
        void apply_settings(gui::Settings const& s);
        gui::Settings current_settings();
        std::vector<std::string> selected_ids() const;

        void start_run(std::string const& target);
        void set_running(bool running);
        void reset_results();

        // Session callbacks, always executed on the UI thread by post().
        void on_begin(std::string const& id, std::string const& label);
        void on_round(int round, int rounds);
        void on_progress(int done, int total);
        void on_result(cst::TestOutcome const& outcome);
        void on_finished(int exit_code);

        void append_log(std::string const& text);
        void set_status(std::wstring const& text);
        void update_progress(double value);
        void post(std::function<void()> fn);

        std::vector<TestRow> rows_;
        std::shared_ptr<std::atomic<bool>> alive_ = std::make_shared<std::atomic<bool>>(true);
        std::atomic<bool> cancel_{ false };
        std::atomic<bool> running_{ false };
        std::wstring current_label_;
        std::string log_buffer_;             // UI thread only
        int tests_done_ = 0;
        int tests_total_ = 0;
        int failures_ = 0;
    };
}

namespace winrt::CpuStressTestGui::factory_implementation
{
    // Required by the C++/WinRT generated activation factory (MainWindow is a
    // runtime class in Project.idl).
    struct MainWindow : MainWindowT<MainWindow, implementation::MainWindow>
    {
    };
}
