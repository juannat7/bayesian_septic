import argparse
 
import src.test_scripts as scripts


if __name__ == '__main__':
    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--method", help="command line method")
    parser.add_argument("-o", "--model_name", help="name of model, should match the file in /configs/")
    parser.add_argument("-d", "--if_dev", help="if run at dev env., if yes, program will clean cache and temporary files if needed")
    parser.add_argument("-mo", "--mod", help="sub-model to test, if not given, the program will run all models at once")
    args = parser.parse_args()
    
    kwargs = vars(args)
    
    getattr(scripts, args.method)(**kwargs)
