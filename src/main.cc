#include <memory>
#include <stdexcept>
#include <torch/torch.h>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <sstream>
#include <args.hxx>

#include "command/command.h"

int main(int argc, const char ** argv) {
    args::ArgumentParser parser("This is a test program.");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    
    args::Command sample(parser, "sample", "Sample from the model using the specified checkpoint directory", [&](args::Subparser &subparser){
        args::ValueFlag<std::filesystem::path> checkpoint(subparser, "checkpoint", "Path to checkpoint directory", {"checkpoint"}, args::Options::Required);
        args::HelpFlag help(subparser, "help", "Display help information", {'h', "help"});

        subparser.Parse();
        command::do_sample(args::get(checkpoint));
    });

    args::Command train(parser, "train", "Train the model with the specified config file", [&](args::Subparser &subparser){
        args::ValueFlag<std::filesystem::path> config(subparser, "config", "Path to config file", {"config"}, args::Options::Required);
        args::Flag show_help(subparser, "help", "Display help information", {"help"});

        subparser.Parse();
        command::do_train(args::get(config));
    });

    try {
        parser.ParseCLI(argc, argv);
    } catch (const args::Help&) {
        std::cout << parser;
        return 0;
    } catch (const args::Error& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }

    return 0;
}

