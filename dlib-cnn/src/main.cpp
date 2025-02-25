#include <dlib/dnn.h>
#include <iostream>
#include <blt/parse/argparse_v2.h>
#include <blt/std/logging.h>
#include <dlib/data_io.h>
#include <dlib/image_transforms.h>
#include <dlib/dir_nav.h>
#include <iterator>
#include <thread>
#include <nn/config.h>

using namespace dlib;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using res       = relu<residual<block,N,bn_con,SUBNET>>;
template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using res_down  = relu<residual_down<block,N,bn_con,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using level1 = res<512,res<512,res_down<512,SUBNET>>>;
template <typename SUBNET> using level2 = res<256,res<256,res<256,res<256,res<256,res_down<256,SUBNET>>>>>>;
template <typename SUBNET> using level3 = res<128,res<128,res<128,res_down<128,SUBNET>>>>;
template <typename SUBNET> using level4 = res<64,res<64,res<64,SUBNET>>>;

template <typename SUBNET> using alevel1 = ares<512,ares<512,ares_down<512,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<256,ares<256,ares<256,ares<256,ares<256,ares_down<256,SUBNET>>>>>>;
template <typename SUBNET> using alevel3 = ares<128,ares<128,ares<128,ares_down<128,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<64,ares<64,ares<64,SUBNET>>>;

using net_type = loss_multiclass_log<fc<3,avg_pool_everything<
                            level1<
                            level2<
                            level3<
                            level4<
                            max_pool<3,3,2,2,relu<bn_con<con<64,7,7,2,2,
                            input_rgb_image_sized<nn::NETWORK_IMAGE_SIZE>
                            >>>>>>>>>>>;

// testing network type (replaced batch normalization with fixed affine transforms)
using anet_type = loss_multiclass_log<fc<3,avg_pool_everything<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<64,7,7,2,2,
                            input_rgb_image_sized<227>
                            >>>>>>>>>>>;


auto create_training_network()
{

}

auto create_test_network()
{


}

void test(int argc, const char* argv[])
{
    using namespace blt::argparse;
    argument_parser_t parser;
    parser.with_help();

    parser.add_flag("-a").set_action(action_t::STORE_TRUE).set_help("This is a really long test string which should create a multi-line condition");
    parser.add_flag("--deep").set_action(action_t::STORE_FALSE);
    parser.add_flag("-b", "--combined").set_action(action_t::STORE_CONST).set_const(50);
    parser.add_flag("--append").set_action(action_t::APPEND).as_type<int>();
    parser.add_flag("--store_choice").set_action(action_t::STORE).as_type<int>().set_choices(1,2,3,4,5).set_metavar("CHOICE");
    parser.add_flag("--required").set_required(true);
    parser.add_flag("--default").set_default("I am a default value");
    parser.add_flag("-t").set_action(action_t::APPEND_CONST).set_dest("test").set_const(5);
    parser.add_flag("-g").set_action(action_t::APPEND_CONST).set_dest("test").set_const(10);
    parser.add_flag("-e").set_action(action_t::APPEND_CONST).set_dest("test").set_const(15);
    parser.add_flag("-f").set_action(action_t::APPEND_CONST).set_dest("test").set_const(20);
    parser.add_flag("-d").set_action(action_t::APPEND_CONST).set_dest("test").set_const(25);
    parser.add_flag("--end").set_action(action_t::EXTEND).set_dest("wow").as_type<float>();
    parser.add_positional("path_with_choices").set_choices("Hello", "World", "Goodbye");

    auto args = parser.parse(argc, argv);
}

int main(int argc, const char* argv[])
{
    // test(argc, argv);
    // auto parser = blt::arg_parse{};
    // parser.addArgument(blt::arg_builder{"model_path"}.setHelp("Model file location").build());
    // parser.addArgument(blt::arg_builder{"image_path"}.setHelp("Path to images - all images inside any subdirectory (recursive) will be considered").build());
    blt::argparse::argument_parser_t parser;
    parser.with_help();
    const auto subparser = parser.add_subparser("mode");

    const auto test_mode = subparser->add_parser("test");

    test_mode->add_flag("--test").make_flag();

    const auto train_mode = subparser->add_parser("train");
    train_mode->add_positional("model_path");
    train_mode->add_positional("image_path");

    auto args = parser.parse(argc, argv);

    if (args.get("mode") == "test")
    {
        if (args.get<bool>("--test"))
            blt::argparse::detail::test();
        else
        {
            const char* arg[] = {"./program", "--help"};
            test(2, arg);
        }
    } else if (args.get("mode") == "train")
    {

    }

}
