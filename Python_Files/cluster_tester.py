#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import healpy as hp
from matplotlib import pyplot as plt
from healpy.newvisufunc import projview, newprojplot


# In[5]:


nside = 64
npix = hp.nside2npix(nside)

m = np.arange(npix)

hp.projview(m, graticule=True, graticule_labels=True)

plt.savefig('./healpy_tester.png',dpi=300)

