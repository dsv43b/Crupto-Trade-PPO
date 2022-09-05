import numpy as np
from PyEMD import EEMD, CEEMDAN
from multiprocessing import Pool
from typing import Union, List, Tuple
from tqdm.notebook import tqdm


def ceemdan_fun(a: int) -> Union[np.array, str]:
    eIMFs_shape = 0
    additional_coefficient = 0
    while eIMFs_shape != 4:
        eIMFs = eemd.ceemdan(test_dd[a-(96+additional_coefficient):a], max_imf=3)
        eIMFs_shape = eIMFs.shape[0]
        additional_coefficient += 1
    return eIMFs[:, -96:], str(df_ALL_train['Full_time'][a])


if __name__ == '__main__':
    test_dd = df_ALL_train['Close_3'].to_numpy()
    start = 96
    end = test_dd.shape[0]

    eemd = CEEMDAN(trials=50)
    eemd.extrema_detection="parabol"
    eemd.noise_seed(10)

    with Pool(4) as pool:
        result = list(tqdm(pool.imap(ceemdan_fun, range(start, end)), total=len(range(start, end))))

    result = sorted(result, key=lambda x: x[1])
    np.save('/content/drive/MyDrive/Colab Notebooks/Торговля криптовалютой/Close_3_ceemdan',
            np.array([i[0] for i in result]))