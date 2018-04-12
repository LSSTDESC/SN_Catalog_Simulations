import yaml
import argparse
from SN_Cosmo import SN_Cosmo
from SN_Sim import SN_Sim
from SN_Ana import SN_Ana

parser = argparse.ArgumentParser(description='Run a SN simulation from a configuration file')
parser.add_argument('config_filename', help='Configuration file in YAML format.')

def run(config_filename):
    # YAML input file.
    config = yaml.load(open(config_filename))
    print(config)
    """
    print(config['SN parameters']['c'])
    if config['SN parameters']['c'] == 'random':
        print('yes')
    """
    exec(config['Simulator']+'('+str(config)+')')


def main(args):
    print('running')
    run(args.config_filename)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
