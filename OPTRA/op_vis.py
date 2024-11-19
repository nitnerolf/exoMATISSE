from astropy.stats import sigma_clip
import numpy as np
import matplotlib.pyplot as plt

def op_extract_simplevis2(cfdata, verbose=True, plot=False):
    print('Extracting visibility')
    psd        = np.abs(cfdata['CF']['CF'])**2
    sumPSD  = np.sum(psd, axis=1)
    
    background = np.abs(cfdata['CF']['bckg'])**2
    
    print('Shape of sumPSD:', sumPSD.shape)
    print('Shape of background:', background.shape)
    
    sumBkg  = np.sum(background, axis=0)
    
    # plt.figure()
    # plt.imshow(sumBkg, cmap='viridis')
    # plt.show()
    
    avgBkg = []
    count = 0
    filtBkg = np.ma.array(sumBkg, mask=sumBkg==0)
    for iwlen in np.arange(sumBkg.shape[0]):
        count+=1
        if count%100 == 0:
            print('iwlen:', iwlen)
        #filtBkg = sigma_clip(sumBkg, sigma=3, maxiters=1)
        vgbk = np.sum(filtBkg[iwlen, :])
        cnt = filtBkg[iwlen, :].count()
        origCnt = len(filtBkg[iwlen, :])
        
        print('vgbk:', vgbk)
        avgBkg.append(vgbk / 5 * (origCnt / cnt)**2)
    
    avgBkg = np.array(avgBkg)
    
    simplevis = 36 / 4 * (sumPSD - avgBkg) / sumPSD[0,:][None, :]
    simplevis[0,:] /= 9
    
    mask = simplevis[0,:] < 0.95 * np.median(simplevis[0,:])
    print('median:', np.median(simplevis[0,:]))
    print('mask:', mask)
    simplevis2 = np.ma.array(simplevis, mask=np.repeat(mask, 7))
       
    plt.figure()
    plt.plot(mask)
    plt.plot(simplevis[0,:])
    plt.show()
    
    return simplevis2

def op_correct_balance_simplevis2(cfdata, verbose=True, plot=False):
    toto