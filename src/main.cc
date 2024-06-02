#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "args.hxx"
#include "command/command.h"
#include "tokenizer/tokenizer.h"
#include "torch/torch.h"

int main(int argc, const char **argv) {
  args::ArgumentParser parser("This is a test program.");
  args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});

  args::Command sample(
      parser, "sample",
      "Sample from the model using the specified checkpoint directory",
      [&](args::Subparser &subparser) {
        args::ValueFlag<std::string> checkpoint(
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
        args::ValueFlag<std::string> config(subparser, "config",
                                            "Path to config file", {"config"},
                                            args::Options::Required);
        args::Flag show_help(subparser, "help", "Display help information",
                             {"help"});

        subparser.Parse();
        command::train_model(args::get(config));
      });

  std::unordered_map<std::string, tokenizer::Type> tokenizers_map{
      {"char", tokenizer::Type::Char}, {"bar", tokenizer::Type::BPE}};
  args::Command dataset(
      parser, "prepare", "Build a new dataset",
      [&](args::Subparser &subparser) {
        args::ValueFlag<std::string> url(subparser, "url",
                                         "URL to fetch dataset from", {"url"},
                                         args::Options::Required);
        args::ValueFlag<std::string> out_dir(
            subparser, "out", "Path to directory store the dataset", {"out"},
            args::Options::Required);
        args::MapFlag<std::string, tokenizer::Type> tok(
            subparser, "tokenizer",
            "Tokenizer type to use for building dataset", {"tok"},
            tokenizers_map, args::Options::Required);
        args::Flag show_help(subparser, "help", "Display help information",
                             {"help"});

        subparser.Parse();
        command::prepare_data(args::get(url), args::get(out_dir),
                              args::get(tok));
      });

  try {
    parser.ParseCLI(argc, argv);
  } catch (const args::Help &) {
    std::cout << parser;
    return 0;
  } catch (const args::Error &e) {
    std::cerr << e.what() << std::endl << parser;
    return 1;
  }

  return 0;
}
