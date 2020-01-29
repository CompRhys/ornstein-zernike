import argparse

def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('--table', type=str)

    parser.add_argument('--rho', type=float)
    parser.add_argument('--temp', type=float)
    parser.add_argument('--dt', type=float)
    parser.add_argument('--dr', type=float)

    parser.add_argument('--burn_steps', type=int)
    parser.add_argument('--burn_iter_max', type=int)

    # parser.add_argument('--bulk_part', type=int)

    parser.add_argument('--box_size', type=float)

    parser.add_argument('--bulk_steps', type=int)
    parser.add_argument('--bulk_iter', type=int)


    parser.add_argument('--output', type=str)
    parser.add_argument('--pass_path', type=str)
    parser.add_argument('--fail_path', type=str)

    args = parser.parse_args()

    return args
