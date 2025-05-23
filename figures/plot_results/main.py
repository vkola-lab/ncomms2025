from utils import *
from source_code import fig_2a_2b, fig_2e_2f, fig_2c_2d_efig3, fig3_efig4, fig5, fig6,  efig1, efig2, efig6

def main():
    config = load_config("config.yml")
    
    fig_2a_2b.plot(config)
    fig_2e_2f.plot(config)
    fig_2c_2d_efig3.plot(config)
    fig3_efig4.plot(config)
    fig4.plot(config)
    fig5.plot(config)
    fig6.plot(config)
    efig1.plot(config)
    efig2.plot(config)
    efig6.plot(config)


if __name__ == "__main__":
    main()
