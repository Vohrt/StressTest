#pragma once

// The Windows headers define min()/max() macros that would break
// std::min/std::max/std::clamp in the WinUI sources.
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>
#include <unknwn.h>
#include <hstring.h>
#include <restrictederrorinfo.h>

#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Windows.Graphics.h>
#include <winrt/Microsoft.UI.h>
#include <winrt/Microsoft.UI.Dispatching.h>
#include <winrt/Microsoft.UI.Windowing.h>
#include <winrt/Microsoft.UI.Xaml.h>
#include <winrt/Microsoft.UI.Xaml.Controls.h>
#include <winrt/Microsoft.UI.Xaml.Controls.Primitives.h>
#include <winrt/Microsoft.UI.Xaml.Data.h>
#include <winrt/Microsoft.UI.Xaml.Input.h>
#include <winrt/Microsoft.UI.Xaml.Interop.h>
#include <winrt/Microsoft.UI.Xaml.Markup.h>
#include <winrt/Microsoft.UI.Xaml.Media.h>
#include <winrt/Microsoft.UI.Xaml.Navigation.h>

// The WinUI XAML tooling requires the implementation headers of the XAML types
// to be visible in the precompiled header (XamlTypeInfo.g.cpp asserts on it, and
// the generated App.xaml.g.hpp entry point references implementation::App).
#include "MainWindow.xaml.h"
#include "App.xaml.h"
