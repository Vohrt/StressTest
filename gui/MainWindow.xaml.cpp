#include "pch.h"
#include "MainWindow.xaml.h"
#if __has_include("MainWindow.g.cpp")
#include "MainWindow.g.cpp"
#endif

#include "common.h"   // CpuInfo, cpu_info()
#include "cst.h"      // all_benchmarks(), BenchInfo
#include "session.h"  // cst::run_session, RunConfig, TestOutcome, format_value

#include <winrt/Microsoft.UI.Xaml.Automation.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

using namespace winrt;
using namespace winrt::Microsoft::UI;
using namespace winrt::Microsoft::UI::Xaml;
using namespace winrt::Microsoft::UI::Xaml::Controls;
using namespace winrt::Microsoft::UI::Xaml::Media;

namespace winrt::CpuStressTestGui::implementation
{
namespace {

// ---------------------------------------------------------------------------
// Small UI helpers
// ---------------------------------------------------------------------------

// Column layout shared by the header and every test row, so they line up.
Grid make_row_grid()
{
    Grid grid;
    grid.ColumnSpacing(8);
    auto add = [&](GridLength const& width) {
        ColumnDefinition column;
        column.Width(width);
        grid.ColumnDefinitions().Append(column);
    };
    add(GridLengthHelper::FromPixels(28));                              // selection
    add(GridLengthHelper::FromPixels(86));                              // id
    add(GridLengthHelper::FromValueAndType(1.15, GridUnitType::Star));  // label
    add(GridLengthHelper::FromPixels(104));                             // throughput
    add(GridLengthHelper::FromPixels(54));                              // unit
    add(GridLengthHelper::FromPixels(84));                              // AIDA64 index
    add(GridLengthHelper::FromValueAndType(1.0, GridUnitType::Star));   // status
    return grid;
}

TextBlock make_cell(hstring const& text, double opacity)
{
    TextBlock cell;
    cell.Text(text);
    cell.VerticalAlignment(VerticalAlignment::Center);
    cell.TextTrimming(TextTrimming::CharacterEllipsis);
    if (opacity < 1.0) cell.Opacity(opacity);
    return cell;
}

SolidColorBrush status_brush(uint8_t r, uint8_t g, uint8_t b)
{
    return SolidColorBrush(Microsoft::UI::ColorHelper::FromArgb(255, r, g, b));
}

// A dependency object is "inside" another when it is the object or one of its
// visual-tree descendants.
bool is_within(DependencyObject const& node, DependencyObject const& ancestor)
{
    DependencyObject current = node;
    while (current) {
        if (current == ancestor) return true;
        current = VisualTreeHelper::GetParent(current);
    }
    return false;
}

template <typename T>
bool is_checked(T const& control)
{
    auto state = control.IsChecked();
    return state && state.Value();
}

double number_or(double value, double fallback)
{
    return std::isnan(value) ? fallback : value;
}

// "±1.2%" volatility of a sustained run, from its per-round samples.
std::wstring volatility_suffix(std::vector<double> const& samples)
{
    if (samples.size() < 3) return {};
    double mean = 0.0;
    for (double v : samples) mean += v;
    mean /= static_cast<double>(samples.size());
    if (!(mean > 0.0)) return {};
    double variance = 0.0;
    for (double v : samples) variance += (v - mean) * (v - mean);
    variance /= static_cast<double>(samples.size());
    wchar_t buffer[64];
    swprintf_s(buffer, L"（波动 ±%.1f%%）", std::sqrt(variance) / mean * 100.0);
    return buffer;
}

std::wstring index_text(double index)
{
    wchar_t buffer[64];
    swprintf_s(buffer, L"%.1f", index);
    return buffer;
}

} // namespace

MainWindow::MainWindow()
{
    InitializeComponent();

    const cst::CpuInfo& ci = cst::cpu_info();
    wchar_t summary[512];
    swprintf_s(summary, L"CPU: %s  ·  %d 物理 / %d 逻辑核心  ·  ISA: %s",
               to_hstring(ci.brand).c_str(), ci.physical_cores, ci.logical_cores,
               to_hstring(ci.isa_summary()).c_str());
    SysInfoText().Text(summary);

    build_test_rows();

    // Restore the previous session's settings, then hook up the close handler
    // (so that closing without ever touching a control still saves state).
    apply_settings(gui::load_settings());
    Closed({ this, &MainWindow::OnClosed });

#ifdef _DEBUG
    // A Debug build does not reproduce the recorded reference fingerprints
    // (different floating point code generation), so every verified workload
    // would be rejected.  Say so instead of leaving the user guessing.
    append_log("\n注意：当前是 Debug 构建，浮点结果与记录的参考指纹不同，运行时会显示校验失败。\n"
               "      测量请使用 Release 构建（build\\cst.exe）。\n");
    set_status(L"Debug 构建：仅供界面开发；测量请使用 Release 构建。");
#endif
}

// ---------------------------------------------------------------------------
// Test table
// ---------------------------------------------------------------------------

void MainWindow::build_test_rows()
{
    // Header row: same columns as the test rows, built by code so the two agree.
    {
        Grid header = make_row_grid();
        header.Children().Append(make_cell(L"选", 0.65));
        auto id = make_cell(L"测试", 0.65);
        Grid::SetColumn(id, 1);
        header.Children().Append(id);
        auto label = make_cell(L"名称", 0.65);
        Grid::SetColumn(label, 2);
        header.Children().Append(label);
        auto value = make_cell(L"吞吐量", 0.65);
        value.TextAlignment(TextAlignment::Right);
        Grid::SetColumn(value, 3);
        header.Children().Append(value);
        auto unit = make_cell(L"单位", 0.65);
        Grid::SetColumn(unit, 4);
        header.Children().Append(unit);
        auto index = make_cell(L"AIDA64 指数", 0.65);
        index.TextAlignment(TextAlignment::Right);
        Grid::SetColumn(index, 5);
        header.Children().Append(index);
        auto status = make_cell(L"状态", 0.65);
        Grid::SetColumn(status, 6);
        header.Children().Append(status);
        HeaderRow().Children().Append(header);
    }

    const SolidColorBrush idle = status_brush(0x88, 0x88, 0x88);
    for (cst::Benchmark* bench : cst::all_benchmarks()) {
        const cst::BenchInfo info = bench->info();

        TestRow row;
        row.id = info.test;

        Grid line = make_row_grid();
        // A transparent background makes the whole row hit-testable, so tapping
        // anywhere on it toggles the selection.
        line.Background(SolidColorBrush(Microsoft::UI::Colors::Transparent()));
        line.Padding(ThicknessHelper::FromLengths(4, 4, 4, 4));

        CheckBox check;
        check.MinWidth(0);
        check.IsChecked(true);
        // Automation ids make the rows addressable for the UI automation smoke
        // test (tools\gui_smoke.ps1) and for accessibility tooling.
        Automation::AutomationProperties::SetAutomationId(check, to_hstring("Select_" + info.test));
        row.check = check;
        line.Children().Append(check);

        auto id = make_cell(to_hstring(info.test), 1.0);
        Grid::SetColumn(id, 1);
        line.Children().Append(id);

        auto label = make_cell(to_hstring(info.label), 1.0);
        Grid::SetColumn(label, 2);
        line.Children().Append(label);

        auto value = make_cell(L"—", 1.0);
        value.TextAlignment(TextAlignment::Right);
        Automation::AutomationProperties::SetAutomationId(value, to_hstring("Value_" + info.test));
        row.value = value;
        Grid::SetColumn(value, 3);
        line.Children().Append(value);

        auto unit = make_cell(to_hstring(info.unit), 0.7);
        Grid::SetColumn(unit, 4);
        line.Children().Append(unit);

        auto index = make_cell(L"—", 1.0);
        index.TextAlignment(TextAlignment::Right);
        Automation::AutomationProperties::SetAutomationId(index, to_hstring("Index_" + info.test));
        row.index = index;
        Grid::SetColumn(index, 5);
        line.Children().Append(index);

        auto status = make_cell(L"未运行", 0.8);
        status.Foreground(idle);
        Automation::AutomationProperties::SetAutomationId(status, to_hstring("Status_" + info.test));
        row.status = status;
        Grid::SetColumn(status, 6);
        line.Children().Append(status);

        const size_t position = rows_.size();
        line.Tapped([this, position](IInspectable const&,
                                     Microsoft::UI::Xaml::Input::TappedRoutedEventArgs const& e) {
            if (position >= rows_.size()) return;
            TestRow& row = rows_[position];
            if (!row.check) return;
            // Taps that landed on the check box itself are handled by the box.
            auto source = e.OriginalSource().try_as<DependencyObject>();
            if (source && is_within(source, row.check)) return;
            row.check.IsChecked(!is_checked(row.check));
        });

        rows_.push_back(row);
        TestRowsPanel().Children().Append(line);
    }
}

std::vector<std::string> MainWindow::selected_ids() const
{
    std::vector<std::string> ids;
    for (TestRow const& row : rows_) {
        if (row.check && is_checked(row.check)) ids.push_back(row.id);
    }
    return ids;
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

void MainWindow::OnRunSelected(IInspectable const&, RoutedEventArgs const&)
{
    std::string target;
    for (std::string const& id : selected_ids()) {
        if (!target.empty()) target += ',';
        target += id;
    }
    start_run(target);
}

void MainWindow::OnRunAll(IInspectable const&, RoutedEventArgs const&)
{
    start_run("all");
}

void MainWindow::OnStop(IInspectable const&, RoutedEventArgs const&)
{
    if (!running_.load()) return;
    cancel_.store(true);
    set_status(L"正在停止…（等待当前一轮结束）");
}

void MainWindow::OnSelectAll(IInspectable const&, RoutedEventArgs const&)
{
    for (TestRow& row : rows_) {
        if (row.check) row.check.IsChecked(true);
    }
}

void MainWindow::OnSelectNone(IInspectable const&, RoutedEventArgs const&)
{
    for (TestRow& row : rows_) {
        if (row.check) row.check.IsChecked(false);
    }
}

void MainWindow::OnModeChanged(IInspectable const&, RoutedEventArgs const&)
{
    // This can fire while the window is still being parsed, so the other
    // controls may not exist yet.
    if (!SecondsBox() || !MinutesBox()) return;
    const bool sustained = is_checked(SustainedRadio());
    SecondsBox().IsEnabled(!running_.load() && !sustained);
    MinutesBox().IsEnabled(!running_.load() && sustained);
}

void MainWindow::OnCsvToggled(IInspectable const&, RoutedEventArgs const&)
{
    if (!CsvPathBox()) return;
    CsvPathBox().IsEnabled(!running_.load() && is_checked(CsvCheck()));
}

void MainWindow::OnClosed(IInspectable const&, WindowEventArgs const&)
{
    gui::save_settings(current_settings());
    alive_->store(false);
    cancel_.store(true);
}

// ---------------------------------------------------------------------------
// Running
// ---------------------------------------------------------------------------

void MainWindow::start_run(std::string const& target)
{
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) return;
    if (target.empty()) {
        running_.store(false);
        set_status(L"没有选中任何测试：请勾选要运行的测试，或点击“全部运行”。");
        return;
    }

    cst::RunConfig config;
    config.target = target;
    config.sustained = is_checked(SustainedRadio());
    config.seconds = number_or(SecondsBox().Value(), 15.0);
    config.minutes = number_or(MinutesBox().Value(), 20.0);
    config.threads = static_cast<int>(number_or(ThreadsBox().Value(), 0.0));
    config.accept_reference = is_checked(AcceptRefCheck());
    config.no_verify = is_checked(NoVerifyCheck());
    if (is_checked(CsvCheck())) config.csv_path = to_string(CsvPathBox().Text());

    cancel_.store(false);
    reset_results();
    tests_done_ = 0;
    tests_total_ = 0;
    failures_ = 0;
    update_progress(0.0);
    set_running(true);

    // The session runs on its own thread; everything it reports is marshalled
    // back to the UI thread through the window's dispatcher queue.
    std::thread([this, config] {
        cst::SessionHooks hooks;
        hooks.log = [this](std::string const& text) {
            post([this, text] { append_log(text); });
        };
        hooks.warn = [this](std::string const& text) {
            post([this, text] { append_log(text); });
        };
        hooks.begin = [this](std::string const& id, std::string const& label) {
            post([this, id, label] { on_begin(id, label); });
        };
        hooks.round = [this](int round, int rounds) {
            post([this, round, rounds] { on_round(round, rounds); });
        };
        hooks.progress = [this](int done, int total) {
            post([this, done, total] { on_progress(done, total); });
        };
        hooks.result = [this](cst::TestOutcome const& outcome) {
            post([this, outcome] {
                if (outcome.rejected) ++failures_;
                on_result(outcome);
            });
        };
        hooks.cancel = [this] { return cancel_.load(); };

        const int code = cst::run_session(config, hooks);
        post([this, code] { on_finished(code); });
    }).detach();
}

void MainWindow::on_begin(std::string const& id, std::string const& label)
{
    current_label_ = to_hstring(label);
    set_status(L"正在运行 " + current_label_ + L" …");
    for (TestRow& row : rows_) {
        if (row.id != id) continue;
        if (row.status) {
            row.status.Text(L"运行中…");
            row.status.Foreground(status_brush(0x15, 0x65, 0xC0));
        }
        if (row.value) row.value.Text(L"…");
        if (row.index) row.index.Text(L"…");
        break;
    }
}

void MainWindow::on_round(int round, int rounds)
{
    if (!running_.load()) return;
    std::wstring text = L"正在运行 " + current_label_;
    if (rounds > 1) {
        wchar_t suffix[64];
        swprintf_s(suffix, L" … 第 %d/%d 轮", round, rounds);
        text += suffix;
    }
    set_status(text);
    if (tests_total_ > 0 && rounds > 0) {
        update_progress((tests_done_ + static_cast<double>(round - 1) / rounds) / tests_total_);
    }
}

void MainWindow::on_progress(int done, int total)
{
    tests_done_ = done;
    tests_total_ = total;
    update_progress(total > 0 ? static_cast<double>(done) / total : 0.0);
}

void MainWindow::on_result(cst::TestOutcome const& outcome)
{
    for (TestRow& row : rows_) {
        if (row.id != outcome.id) continue;
        if (row.value) row.value.Text(to_hstring(cst::format_value(outcome.value, outcome.unit)));
        if (row.index) row.index.Text(index_text(outcome.index));
        if (row.status) {
            if (outcome.rejected) {
                row.status.Text(outcome.digest_unstable ? L"✗ 输出不稳定（得分按 0 计）"
                                                        : L"✗ 校验失败（得分按 0 计）");
                row.status.Foreground(status_brush(0xC6, 0x28, 0x28));
            } else if (outcome.reference_missing) {
                row.status.Text(L"· 无参考记录（可勾选“记录本次为参考”）");
                row.status.Foreground(status_brush(0xEF, 0x6C, 0x00));
            } else if (outcome.reference_params_changed) {
                row.status.Text(L"· 参数与参考不同，不可比");
                row.status.Foreground(status_brush(0xEF, 0x6C, 0x00));
            } else {
                row.status.Text((outcome.has_full_digest ? L"✓ 校验通过" : L"✓ 完成") +
                                volatility_suffix(outcome.samples));
                row.status.Foreground(status_brush(0x2E, 0x7D, 0x32));
            }
        }
        break;
    }
}

void MainWindow::on_finished(int exit_code)
{
    // A cancel that arrived after the last workload finished still reports as a
    // cancellation; treat a fully completed run as success in that case.
    running_.store(false);
    set_running(false);
    update_progress(1.0);

    switch (exit_code) {
    case 0:
        set_status(L"完成：所有测试通过校验。");
        break;
    case 3:
        set_status(L"完成：有 " + std::to_wstring(failures_) +
                   L" 项未通过校验（分数已按 0 记录，详见日志）。");
        break;
    case 4:
        set_status(L"已停止。");
        break;
    case 2:
        set_status(L"错误：测试目标无效。");
        break;
    default:
        set_status(L"结束（退出码 " + std::to_wstring(exit_code) + L"）。");
        break;
    }
}

void MainWindow::reset_results()
{
    const SolidColorBrush idle = status_brush(0x88, 0x88, 0x88);
    for (TestRow& row : rows_) {
        if (row.value) row.value.Text(L"—");
        if (row.index) row.index.Text(L"—");
        if (row.status) {
            row.status.Text(L"未运行");
            row.status.Foreground(idle);
        }
    }
}

void MainWindow::set_running(bool running)
{
    running_.store(running);
    const bool sustained = is_checked(SustainedRadio());
    RunButton().IsEnabled(!running);
    RunAllButton().IsEnabled(!running);
    SelectAllButton().IsEnabled(!running);
    SelectNoneButton().IsEnabled(!running);
    StopButton().IsEnabled(running);
    SingleRadio().IsEnabled(!running);
    SustainedRadio().IsEnabled(!running);
    AcceptRefCheck().IsEnabled(!running);
    NoVerifyCheck().IsEnabled(!running);
    CsvCheck().IsEnabled(!running);
    ThreadsBox().IsEnabled(!running);
    SecondsBox().IsEnabled(!running && !sustained);
    MinutesBox().IsEnabled(!running && sustained);
    CsvPathBox().IsEnabled(!running && is_checked(CsvCheck()));
}

void MainWindow::update_progress(double value)
{
    if (value < 0.0) value = 0.0;
    if (value > 1.0) value = 1.0;
    RunProgress().Value(value);
}

void MainWindow::set_status(std::wstring const& text)
{
    StatusText().Text(text);
}

void MainWindow::append_log(std::string const& text)
{
    log_buffer_ += text;
    // Keep the buffer bounded: a long sustained run would otherwise grow it
    // without limit.  Old text goes first, so the newest output stays visible.
    constexpr size_t kLimit = 1u << 20;
    if (log_buffer_.size() > kLimit) {
        log_buffer_.erase(0, log_buffer_.size() - kLimit / 2);
        log_buffer_.insert(0, "... (earlier output trimmed) ...\n");
    }
    LogText().Text(to_hstring(log_buffer_));
    LogScroll().UpdateLayout();
    LogScroll().ChangeView(nullptr, LogScroll().ScrollableHeight(), nullptr);
}

void MainWindow::post(std::function<void()> fn)
{
    auto queue = DispatcherQueue();
    if (!queue) return;
    queue.TryEnqueue([fn] {
        if (!fn) return;
        fn();
    });
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

void MainWindow::apply_settings(gui::Settings const& settings)
{
    SecondsBox().Value(settings.seconds);
    MinutesBox().Value(settings.minutes);
    ThreadsBox().Value(settings.threads);
    SustainedRadio().IsChecked(settings.sustained);
    SingleRadio().IsChecked(!settings.sustained);
    AcceptRefCheck().IsChecked(settings.accept_reference);
    NoVerifyCheck().IsChecked(settings.no_verify);
    CsvCheck().IsChecked(settings.csv);
    CsvPathBox().Text(to_hstring(settings.csv_path));

    if (settings.all_selected) {
        for (TestRow& row : rows_) {
            if (row.check) row.check.IsChecked(true);
        }
    } else {
        for (TestRow& row : rows_) {
            if (!row.check) continue;
            bool on = false;
            for (std::string const& id : settings.selected) {
                if (id == row.id) { on = true; break; }
            }
            row.check.IsChecked(on);
        }
    }

    if (settings.window_width >= 640 && settings.window_height >= 480) {
        AppWindow().Resize(winrt::Windows::Graphics::SizeInt32{ settings.window_width,
                                                                settings.window_height });
    }

    OnModeChanged(nullptr, nullptr);
    OnCsvToggled(nullptr, nullptr);
}

gui::Settings MainWindow::current_settings()
{
    gui::Settings settings;
    settings.seconds = number_or(SecondsBox().Value(), 15.0);
    settings.minutes = number_or(MinutesBox().Value(), 20.0);
    settings.threads = static_cast<int>(number_or(ThreadsBox().Value(), 0.0));
    settings.sustained = is_checked(SustainedRadio());
    settings.accept_reference = is_checked(AcceptRefCheck());
    settings.no_verify = is_checked(NoVerifyCheck());
    settings.csv = is_checked(CsvCheck());
    settings.csv_path = to_string(CsvPathBox().Text());

    std::vector<std::string> selected = selected_ids();
    settings.all_selected = (selected.size() == rows_.size());
    if (!settings.all_selected) settings.selected = selected;

    const auto size = AppWindow().Size();
    settings.window_width = size.Width;
    settings.window_height = size.Height;
    return settings;
}

} // namespace winrt::CpuStressTestGui::implementation
