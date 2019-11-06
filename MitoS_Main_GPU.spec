# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['MitoS_Main_GPU.py'],
             pathex=['C:\\Users\\Christian\\Documents\\GitHub_MSN\\MitoS_segmentation_tool'],
             binaries=[],
             datas=[('C:\\Users\\Christian\\Anaconda3\\lib\\site-packages\\dask\\dask.yaml', '.\\dask')],
             hiddenimports=["pywt","pywt._extensions._cwt"],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='MitoS_Main_GPU',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
