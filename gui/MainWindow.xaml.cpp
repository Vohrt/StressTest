#include "pch.h"
#include "MainWindow.xaml.h"
#if __has_include("MainWindow.g.cpp")
#include "MainWindow.g.cpp"
#endif

#include "common.h"   // CpuInfo, cpu_info()
#include "cst.h"      // all_benchmarks(), BenchInfo
#include "session.h"  // cst::run_session, RunConfig, TestOutcome, format_value

#include <winrt/Microsoft.UI.Interop.h>
#include <winrt/Microsoft.UI.Xaml.Automation.h>

#include <algorithm>
#include <atomic>
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

// The control rows are two non-wrapping StackPanels, so the window must not go
// below the width they need - otherwise the run controls (and the CSV row) are
// clipped with no way to scroll to them.  The width is measured after the first
// layout (see apply_minimum_size) and converted to the physical pixels the
// AppWindow APIs use; these are the effective-pixel fallbacks used until then.
constexpr double kFallbackToolbarWidth = 1140.0;  // effective px, measured on this host
constexpr double kMinContentHeight = 430.0;       // effective px
constexpr int kTitleBarAllowance = 40;            // effective px, plus frame

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

// The verdict of a sustained run in the GUI's language; the CLI prints its own
// English text (cst::sustain_verdict_text).
const wchar_t* verdict_text(cst::SustainVerdict verdict)
{
    switch (verdict) {
    case cst::SustainVerdict::Throttling: return L"降频";
    case cst::SustainVerdict::MildDecay: return L"轻微衰减";
    case cst::SustainVerdict::Increasing: return L"上升（异常）";
    default: return L"稳定";
    }
}

// "（波动 ±0.6%，漂移 -0.26%，稳定）" appended to a sustained run's status.
// Empty for a single measurement: fewer than three samples have no drift story.
std::wstring sustain_suffix(std::vector<double> const& samples)
{
    const cst::SustainSummary s = cst::summarize_samples(samples);
    if (!s.enough) return {};
    wchar_t buffer[160];
    swprintf_s(buffer, L"（波动 ±%.1f%%，漂移 %+.2f%%，%s）", s.sd_percent, s.drift_percent,
               verdict_text(s.verdict));
    return buffer;
}

// The rest of the sustained picture as a tool tip on the status cell, so the
// numbers the log prints (mean, p5/p95) are reachable without scrolling it.
std::wstring sustain_tooltip(cst::TestOutcome const& outcome)
{
    const cst::SustainSummary s = cst::summarize_samples(outcome.samples);
    if (!s.enough) return {};
    wchar_t buffer[256];
    swprintf_s(buffer,
               L"样本 %zu 个 · 均值 %s %s（±%.2f%%）\np5 / p95：%s / %s %s\n"
               L"漂移（首/末 1/3）：%+.2f%% → %s",
               s.count, to_hstring(cst::format_value(s.mean, outcome.unit)).c_str(),
               to_hstring(outcome.unit).c_str(), s.sd_percent,
               to_hstring(cst::format_value(s.p5, outcome.unit)).c_str(),
               to_hstring(cst::format_value(s.p95, outcome.unit)).c_str(),
               to_hstring(outcome.unit).c_str(), s.drift_percent, verdict_text(s.verdict));
    return buffer;
}

// The AIDA64 index column.  Without a real calibration entry (a row in
// calibration.txt with an AIDA64 reference score) the "index" would just repeat
// the raw throughput, so say 未校准 instead of dressing it up as a score.
std::wstring index_text(cst::TestOutcome const& outcome)
{
    if (outcome.rejected) return L"0.0";
    if (!outcome.calibrated) return L"未校准";
    wchar_t buffer[64];
    swprintf_s(buffer, L"%.1f", outcome.index);
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

    // The thread count cannot usefully exceed the logical core count, and the
    // per-thread working sets (96 MiB per worker in the memory tests) multiply
    // by it - an extreme value used to be able to kill the process.  Cap the
    // input itself so the spinner cannot even suggest otherwise.
    ThreadsBox().Minimum(0.0);
    ThreadsBox().Maximum(static_cast<double>(std::max(1, ci.logical_cores)));

    // Pin the window minimum before restoring the saved size, so a size saved
    // by an older version (or on a different monitor) cannot bring the clipped
    // toolbars back.  OverlappedPresenter exposes the minimum as two
    // IReference<int32_t> properties; passing an int uses IReference's
    // converting constructor.
    apply_minimum_size();

    // Restore the previous session's settings, then hook up the close handler
    // (so that closing without ever touching a control still saves state).
    apply_settings(gui::load_settings());
    Closed({ this, &MainWindow::OnClosed });

    // The toolbars' real width is only known after the first layout pass (and
    // it changes with the monitor's DPI scale), so refine the minimum then and
    // grow the window if the restored size is too small.
    if (auto content = Content().try_as<FrameworkElement>()) {
        content.Loaded([this](IInspectable const&, RoutedEventArgs const&) {
            ensure_minimum_size();
            if (!root_changed_hooked_) {
                if (auto root = Content().XamlRoot()) {
                    root.Changed([this](XamlRoot const&, XamlRootChangedEventArgs const&) {
                        ensure_minimum_size();
                    });
                    root_changed_hooked_ = true;
                }
            }
        });
    }

    // Settings used to be written only when the window closed, so a crash or a
    // forced kill lost them.  Save the easiest thing to lose - a resize or a
    // move - as it happens; the run controls are saved when a run starts.
    AppWindow().Changed(
        [this](Microsoft::UI::Windowing::AppWindow const&,
               Microsoft::UI::Windowing::AppWindowChangedEventArgs const& args) {
            if (args.DidSizeChange() || args.DidPositionChange()) {
                gui::save_settings(current_settings());
            }
        });

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
    cancel_state_->store(true);
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

// Recording a new golden reference overwrites the baseline every future run is
// checked against, so it is a one-shot action with an explicit confirmation -
// never a check box that a later session would find already ticked.  It is not
// written to settings.ini either.
fire_and_forget MainWindow::OnRecordReference(IInspectable const&, RoutedEventArgs const&)
{
    if (running_.load()) co_return;

    std::string target;
    for (std::string const& id : selected_ids()) {
        if (!target.empty()) target += ',';
        target += id;
    }
    if (target.empty()) {
        set_status(L"没有选中任何测试：请先勾选要记录为参考的测试。");
        co_return;
    }

    ContentDialog dialog;
    dialog.XamlRoot(Content().XamlRoot());
    dialog.Title(box_value(L"记录本次为参考？"));
    dialog.PrimaryButtonText(L"运行并记录");
    dialog.CloseButtonText(L"取消");
    dialog.DefaultButton(ContentDialogButton::Close);
    TextBlock body;
    body.TextWrapping(TextWrapping::Wrap);
    body.MaxWidth(460);
    body.Text(L"将运行选中的测试，并把它们的输出指纹写入 reference.txt：\n\n"
              L"· 只影响这次运行，之后不会自动重复；\n"
              L"· 会覆盖这些测试已有的参考指纹；\n"
              L"· 请只在确认系统稳定时记录。\n\n"
              L"如果 CPU 在不稳定状态下被记录，坏指纹会成为基线，"
              L"之后正常的结果反而会被判为失败。");
    dialog.Content(body);

    ContentDialogResult result = ContentDialogResult::None;
    try {
        result = co_await dialog.ShowAsync();
    } catch (...) {
        co_return;   // a dialog that cannot be shown must not start the run
    }
    if (result != ContentDialogResult::Primary) co_return;
    start_run(target, /*record_reference=*/true);
}

void MainWindow::OnClosed(IInspectable const&, WindowEventArgs const&)
{
    gui::save_settings(current_settings());
    alive_->store(false);
    cancel_state_->store(true);
}

// ---------------------------------------------------------------------------
// Running
// ---------------------------------------------------------------------------

void MainWindow::start_run(std::string const& target, bool record_reference)
{
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) return;
    if (target.empty()) {
        running_.store(false);
        set_status(record_reference
                       ? L"没有选中任何测试：请先勾选要记录为参考的测试。"
                       : L"没有选中任何测试：请勾选要运行的测试，或点击“全部运行”。");
        return;
    }

    // Save what the user is looking at before the run starts; settings used to
    // be written only on close, so a crash lost everything since the last one.
    gui::save_settings(current_settings());

    cst::RunConfig config;
    config.target = target;
    config.sustained = is_checked(SustainedRadio());
    config.seconds = number_or(SecondsBox().Value(), 15.0);
    config.minutes = number_or(MinutesBox().Value(), 20.0);
    config.threads = static_cast<int>(number_or(ThreadsBox().Value(), 0.0));
    // Both verification options are one-shot for this run only: the record
    // flag comes from the confirmation dialog, and neither flag is persisted.
    config.accept_reference = record_reference;
    config.no_verify = is_checked(NoVerifyCheck());
    if (is_checked(CsvCheck())) config.csv_path = to_string(CsvPathBox().Text());

    // The number of workloads is known before the session starts, so the
    // progress bar moves from the first round on - the session's own progress
    // hook only fires after a workload has finished.
    tests_total_ = target == "all"
                       ? static_cast<int>(cst::all_benchmarks().size())
                       : static_cast<int>(std::count(target.begin(), target.end(), ',')) + 1;
    tests_done_ = 0;
    failures_ = 0;
    cancel_state_ = std::make_shared<std::atomic<bool>>(false);
    reset_results();
    update_progress(0.0);
    set_running(true);

    // The session runs on its own thread.  That thread never touches the window
    // object itself: `self` is only copied around, and the member calls happen
    // in lambdas that run on the UI thread where `post_local` re-checked
    // `alive` - anything queued after the window closed is dropped.  The cancel
    // flag is a shared_ptr the thread keeps alive on its own.
    auto alive = alive_;
    auto cancel = cancel_state_;
    auto queue = DispatcherQueue();
    MainWindow* self = this;   // copied, never dereferenced on the session thread
    std::thread([self, config, alive, cancel, queue] {
        const auto post_local = [alive, queue](std::function<void()> fn) {
            if (!queue) return;
            queue.TryEnqueue([alive, fn = std::move(fn)] {
                if (!alive->load() || !fn) return;
                fn();
            });
        };

        cst::SessionHooks hooks;
        hooks.log = [self, post_local](std::string const& text) {
            post_local([self, text] { self->append_log(text); });
        };
        hooks.warn = [self, post_local](std::string const& text) {
            post_local([self, text] { self->append_log(text); });
        };
        hooks.begin = [self, post_local](std::string const& id, std::string const& label) {
            post_local([self, id, label] { self->on_begin(id, label); });
        };
        hooks.round = [self, post_local](int round, int rounds) {
            post_local([self, round, rounds] { self->on_round(round, rounds); });
        };
        hooks.progress = [self, post_local](int done, int total) {
            post_local([self, done, total] { self->on_progress(done, total); });
        };
        hooks.result = [self, post_local](cst::TestOutcome const& outcome) {
            post_local([self, outcome] {
                if (outcome.rejected) ++self->failures_;
                self->on_result(outcome);
            });
        };
        hooks.cancel = [cancel] { return cancel->load(); };

        int code = 5;
        try {
            code = cst::run_session(config, hooks);
        } catch (std::exception const& e) {
            const std::string message = e.what();
            post_local([self, message] { self->on_run_error(message); });
            return;
        } catch (...) {
            post_local([self] { self->on_run_error("unknown exception"); });
            return;
        }
        post_local([self, code] { self->on_finished(code); });
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
        if (row.index) {
            row.index.Text(index_text(outcome));
            // Uncalibrated rows say 未校准 instead of a number; dim them so
            // they do not read as a score at a glance.
            row.index.Opacity(outcome.calibrated ? 1.0 : 0.55);
        }
        if (row.status) {
            // Any old sustained tool tip belongs to the previous run.
            ToolTipService::SetToolTip(row.status, nullptr);
            if (outcome.rejected) {
                row.status.Text(outcome.digest_unstable ? L"✗ 输出不稳定（得分按 0 计）"
                                                        : L"✗ 校验失败（得分按 0 计）");
                row.status.Foreground(status_brush(0xC6, 0x28, 0x28));
            } else if (outcome.reference_missing) {
                row.status.Text(L"· 无参考记录（可用“记录为参考…”重录）");
                row.status.Foreground(status_brush(0xEF, 0x6C, 0x00));
            } else if (outcome.reference_params_changed) {
                row.status.Text(L"· 参数与参考不同，不可比");
                row.status.Foreground(status_brush(0xEF, 0x6C, 0x00));
            } else {
                // The sustained verdict goes right next to the row it belongs
                // to; the log pane keeps the full report.
                row.status.Text((outcome.has_full_digest ? L"✓ 校验通过" : L"✓ 完成") +
                                sustain_suffix(outcome.samples));
                const std::wstring tip = sustain_tooltip(outcome);
                if (!tip.empty()) {
                    ToolTipService::SetToolTip(row.status, box_value(hstring(tip)));
                }
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
    case 5:
        set_status(L"错误：运行失败（详见日志）。");
        break;
    default:
        set_status(L"结束（退出码 " + std::to_wstring(exit_code) + L"）。");
        break;
    }
}

// The session threw (allocation failure, ...).  Without this it would have
// killed the process from a detached thread, with no message anywhere.
void MainWindow::on_run_error(std::string const& message)
{
    running_.store(false);
    set_running(false);
    update_progress(1.0);
    append_log("error: run session failed: " + message + "\n");
    const hstring wide = to_hstring(message);
    set_status(std::wstring(L"运行失败：") + std::wstring(wide.c_str()));
    show_error_dialog(wide);
}

fire_and_forget MainWindow::show_error_dialog(hstring message)
{
    ContentDialog dialog;
    dialog.XamlRoot(Content().XamlRoot());
    dialog.Title(box_value(L"运行失败"));
    TextBlock body;
    body.TextWrapping(TextWrapping::Wrap);
    body.MaxWidth(460);
    body.Text(message);
    dialog.Content(body);
    dialog.CloseButtonText(L"确定");
    dialog.DefaultButton(ContentDialogButton::Close);
    try {
        co_await dialog.ShowAsync();
    } catch (...) {
        // Nothing left to report the failure to - the log and the status line
        // already carry it.
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
    RecordRefButton().IsEnabled(!running);
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

// ---------------------------------------------------------------------------
// Minimum window size
// ---------------------------------------------------------------------------

// Effective pixels per physical pixel for the window (1.25 on a 125% display).
// XamlRoot is not available until the first layout, so fall back to the DPI the
// window was created with.
double MainWindow::rasterization_scale()
{
    if (auto content = Content()) {
        if (auto root = content.XamlRoot()) {
            const double scale = root.RasterizationScale();
            if (scale > 0.0) return scale;
        }
    }
    const HWND hwnd = winrt::Microsoft::UI::GetWindowFromWindowId(AppWindow().Id());
    const UINT dpi = hwnd ? GetDpiForWindow(hwnd) : 96u;
    return dpi >= 48u ? dpi / 96.0 : 1.0;
}

// Recomputes the preferred minimum size in the physical pixels AppWindow speaks.
// The toolbars are non-wrapping, so below this width their right-hand controls
// (the thread-count hint, the CSV path, 全选/全不选) would be clipped - which is
// exactly what happened when the minimum was a fixed number that ignored the
// display scale.
void MainWindow::apply_minimum_size()
{
    const double scale = rasterization_scale();
    int width = static_cast<int>(std::ceil(kFallbackToolbarWidth * scale));
    int height = static_cast<int>(std::ceil((kMinContentHeight + kTitleBarAllowance) * scale));

    if (RunRow() && OptionsRow()) {
        const double rows = std::max(RunRow().DesiredSize().Width, OptionsRow().DesiredSize().Width);
        if (rows > 0.0) {
            // The desired width excludes the page's 16 px padding on both sides;
            // a few physical pixels cover the window frame.
            width = static_cast<int>(std::ceil((rows + 32.0) * scale)) + 8;
        }
    }

    min_window_width_ = width;
    min_window_height_ = height;
    if (auto presenter =
            AppWindow().Presenter().try_as<Microsoft::UI::Windowing::OverlappedPresenter>()) {
        presenter.PreferredMinimumWidth(width);
        presenter.PreferredMinimumHeight(height);
    }
}

void MainWindow::ensure_minimum_size()
{
    apply_minimum_size();
    const auto size = AppWindow().Size();
    if (size.Width < min_window_width_ || size.Height < min_window_height_) {
        AppWindow().Resize(winrt::Windows::Graphics::SizeInt32{
            std::max(size.Width, min_window_width_), std::max(size.Height, min_window_height_)});
    }
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

void MainWindow::apply_settings(gui::Settings const& settings)
{
    SecondsBox().Value(settings.seconds);
    MinutesBox().Value(settings.minutes);
    ThreadsBox().Value(std::clamp(settings.threads, 0, static_cast<int>(ThreadsBox().Maximum())));
    SustainedRadio().IsChecked(settings.sustained);
    SingleRadio().IsChecked(!settings.sustained);
    // "记录为参考" and "跳过校验" are deliberately not restored - both are
    // one-shot options (see Settings.h); a stale ini file cannot switch
    // verification off.
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

    // A size saved by an older version (or a hand-edited ini) could be smaller
    // than the toolbars need; clamp instead of restoring it and clipping the UI.
    const int width = std::max(settings.window_width, min_window_width_);
    const int height = std::max(settings.window_height, min_window_height_);
    AppWindow().Resize(winrt::Windows::Graphics::SizeInt32{ width, height });

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
