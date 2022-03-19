import os


# returns the absolute path to main dir of the project
def get_path():
    # get current dir abs path
    cur_dir = os.path.abspath(os.curdir)
    # go back to parent dir
    cur_dir2 = os.path.abspath(os.path.join(cur_dir, os.pardir))
    # go back to subsequent parent dir
    main_dir = os.path.abspath(os.path.join(cur_dir2, os.pardir))
    return main_dir
