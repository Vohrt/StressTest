// Command line front end entry point.
//
// The executable is both the GUI and the CLI: gui/App.xaml.cpp implements
// wWinMain, decides between the two, and calls cst::cli_main for the command
// line.  src/cli_entry.cpp provides a plain main() for the CLI-only build.
#pragma once

namespace cst {

int cli_main(int argc, char** argv);

} // namespace cst
