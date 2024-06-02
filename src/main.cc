#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "args.hxx"
#include "command/command.h"
#include "torch/torch.h"

int main(int argc, const char **argv) {
  args::ArgumentParser parser("This is a test program.");
  args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});

  args::Command sample(
      parser, "sample",
      "Sample from the model using the specified checkpoint directory",
      [&](args::Subparser &subparser) {
        args::ValueFlag<std::filesystem::path> checkpoint(
            subparser, "checkpoint", "Path to checkpoint directory",
            {"checkpoint"}, args::Options::Required);
        args::ValueFlag<std::string> device(
            subparser, "device",
            "Device to run the model on (cpu | cuda | mps, etc.)",
            {"checkpoint"}, args::Options::Required);
        args::HelpFlag help(subparser, "help", "Display help information",
                            {'h', "help"});

        subparser.Parse();
        command::sample_model(args::get(checkpoint),
                              torch::Device{args::get(device)});
      });

  args::Command train(
      parser, "train", "Train the model with the specified config file",
      [&](args::Subparser &subparser) {
        args::ValueFlag<std::filesystem::path> config(
            subparser, "config", "Path to config file", {"config"},
            args::Options::Required);
        args::Flag show_help(subparser, "help", "Display help information",
                             {"help"});

        subparser.Parse();
        command::train_model(args::get(config));
      });

  try {
    parser.ParseCLI(argc, argv);
  } catch (const args::Help &) {
    std::cout << parser;
    return 0;
  } catch (const args::Error &e) {
    std::cerr << e.what() << std::endl;
    std::cerr << parser;
    return 1;
  }

  return 0;
}
