#pragma once

#include "App.xaml.g.h"

namespace winrt::CpuStressTestGui::implementation
{
    struct App : AppT<App>
    {
        App();
        void OnLaunched(winrt::Microsoft::UI::Xaml::LaunchActivatedEventArgs const&);

    private:
        winrt::Microsoft::UI::Xaml::Window window{ nullptr };
    };
}
