set -o xtrace

setup_root() {
    apt-get install -qq -y \
        python3-pip \
        python3-tk

    pip3 install -qq \
        pytest \
        scikit-image \
        scikit-learn \
        matplotlib 
}

setup_checker() {
    python3 -c 'import matplotlib.pyplot'
}

"$@"