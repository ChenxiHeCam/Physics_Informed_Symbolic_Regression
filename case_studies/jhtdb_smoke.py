import numpy as np
TOKEN='uk.ac.cam.ch2067-c731b129'
try:
    from givernylocal.turbulence_dataset import turb_dataset
    from givernylocal.turbulence_toolkit import getData
    ds = turb_dataset(dataset_title='isotropic1024coarse', output_path='./_gv_out', auth_token=TOKEN)
    pts = (np.random.RandomState(0).rand(8,3)*2*np.pi).astype(np.float64)
    r = getData(ds, 'velocity', 0.0, 'none', 'none', 'field', pts)
    print('OK givernylocal shape/type:', type(r))
    arr = np.array(r)
    print('array shape', arr.shape)
    print(arr[:3])
except Exception as e:
    import traceback; traceback.print_exc()
    print('GIVERNY_FAIL', type(e).__name__, str(e)[:200])
