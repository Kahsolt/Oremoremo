#!/usr/bin/env pythonw3
# Author: Armit
# Create Time: 2020/02/08 

import os
import sys
import time
import logging
from configparser import ConfigParser
from threading import Thread, RLock, Event
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.font as tkfont
import tkinter.messagebox as tkmsg
import tkinter.filedialog as tkfiledlg
import shutil
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pyaudio                  # audio record/play
import wave
from scipy.io import wavfile    # wavfile read/write
from pydub import AudioSegment  # wav manipulation
import librosa                  # DSP
import librosa.display as rosadisplay  # wavplot

__version__ = '0.1'

# settings
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
LIST_DIR = os.getenv('LIST_DIR', 'list')
SAVE_DIR = os.getenv('SAVE_DIR', 'result')
TMP_DIR = os.getenv('TMP_DIR', 'tmp')
LOG_FILE = os.getenv('LOG_FILE', None)
CONFIG_FILE = os.getenv('CONFIG_FILE', 'config.ini')
RECLIST_FILE = os.getenv('RECLIST_FILE', 'reclist.txt')
TYPELIST_FILE = os.getenv('TYPELIST_FILE', 'typelist.txt')

# Standard: 44100Hz 16bit Mono
FRAME_RATE = 44100
SAMPLE_WIDTH = 2
CHANNELS = 1
CHUNK = 1024
RECORD_SECONDS = 0.025

WINDOW_TITLE = "OremoRemo (Ver %s)" % __version__
WINDOW_SIZE = (800, 600)
GRAPH_DPI = 64      # smaller clearer

# utils & sysfix
def open(fp, rw='r', **kwargs):
  from builtins import open as _open
  return 'b' in rw and _open(fp, rw, **kwargs) or _open(fp, rw, encoding='utf8', **kwargs)

def fuck_encoding(bdata:bytes) -> str:
  ENCODINGS = ['shiftjis', 'gb18030', 'ascii', 'utf8']
  for encoding in ENCODINGS:
    try: return bdata.decode(encoding)
    except UnicodeDecodeError: pass
  return None

# app
class Config:    # TODO: make all configurables configured

  def __init__(self, fp='config.ini', sep=':'):
    self.fp = os.path.abspath(fp)
    self.cfg = ConfigParser()
    self.sep = sep

    self.default() # load default
    self.load()    # allow user-config overwrite
  
  def __getitem__(self, key, to_type=str):
    sec, opt = key.split(self.sep)
    return self.cfg.has_option(sec, opt) and to_type(self.cfg[sec][opt]) or None

  def __setitem__(self, key, val):
    sec, opt = key.split(self.sep)
    if not self.cfg.has_section(sec): self.cfg.add_section(sec)
    self.cfg[sec][opt] = str(val)

  def default(self):
    self['System:BASE_PATH'] = os.path.dirname(os.path.abspath(__file__))
    self['System:LANGUAGE'] = 'en'

  def load(self):
    if not os.path.exists(self.fp): return
    self.cfg.read(self.fp, encoding='utf8')
  
  def save(self):
    with open(self.fp, 'w+') as fh:
      self.cfg.write(fh)

class App:

  def __init__(self):
    self.setup_logger()
    logger.debug('[%s] initializing' % self.__class__.__name__)

    self.cfg = Config()
    self.T = defaultdict(lambda: '<???>')   # UI translation text res

    p = pyaudio.PyAudio()
    self.stream = p.open(format=p.get_format_from_width(SAMPLE_WIDTH), channels=CHANNELS, rate=FRAME_RATE,
                         input=True, output=True, frames_per_buffer=CHUNK)
    self.thr_record = None                  # record thread
    self.working_record = Event()           # whether recording
    self.thr_listen = None                  # listening thread
    self.working_listen = Event()           # whether testing

    self.records = defaultdict(lambda: ([ ], [ ]))    # { vsymtype<str>: (chunks<[bytes]>, data<ndarray>) }
    self.cur_vsym = ''
    self.cur_vtype = ''

    self.setup_gui()
    self.setup_workspace()

    logger.debug('[%s] ready' % self.__class__.__name__)
    try: tk.mainloop()
    except KeyboardInterrupt: logger.debug('[%s] KeyboardInterrupt' % self.__class__.__name__)
    finally:
      self.stream.close()
      p.terminate()
    logger.debug('[%s] exited' % self.__class__.__name__)

  def setup_logger(self):
    global logger
    logger = logging.getLogger(WINDOW_TITLE)
    logger.setLevel(level=logging.INFO)

    con = logging.StreamHandler()
    con.setLevel(logging.INFO)
    # con.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(con)

    if LOG_FILE:
      lf = logging.FileHandler(os.path.join(BASE_PATH, LOG_FILE), encoding='utf8')
      lf.setLevel(logging.WARN)
      lf.setFormatter(logging.Formatter("%(asctime)s %(levelname)s:%(lineno)s %(message)s"))
      logger.addHandler(lf)

  def setup_gui(self):
    # root window
    wnd = tk.Tk()
    wnd.title(WINDOW_TITLE)
    (wndw, wndh), scrw, scrh = WINDOW_SIZE, wnd.winfo_screenwidth(), wnd.winfo_screenheight()
    wnd.geometry('%dx%d+%d+%d' % (wndw, wndh, (scrw - wndw) // 2, (scrh - wndh) // 4))
    wnd.resizable(False, False)
    wnd.protocol('WM_DELETE_WINDOW', wnd.quit)     # don't know why close window does't exit app
    wnd.bind('<Key-l>', lambda evt: self.audio_listen())
    wnd.bind('<Key-r>', lambda evt: self.audio_record())
    wnd.bind('<space>', lambda evt: self.audio_play())
    self.wnd = wnd

    # font
    ft_label = tkfont.Font(family='Courier New', size=38, weight=tkfont.BOLD)
    ft_list = tkfont.Font(family='Courier New', size=22)
    
    # main menu bar
    menu = tk.Menu(wnd, tearoff=False)
    wnd.config(menu=menu)
    self.menu = menu
    if True:
      sm = tk.Menu(menu, tearoff=False)
      sm.add_command(label="Load reclist...", command=lambda: self.list_load_('symbol'))
      sm.add_command(label="Load typelist...", command=lambda: self.list_load_('type'))
      sm.add_separator()
      sm.add_command(label="Save records", command=self.audio_save)
      sm.add_separator()
      sm.add_command(label="Exit", command=wnd.quit)
      menu.add_cascade(label="File", menu=sm)
      
      sm = tk.Menu(menu, tearoff=False)
      var = tk.BooleanVar(wnd, value=True)
      self.var_fig_wave = var
      sm.add_checkbutton(label="Show Wave", variable=var, command=self._clt_fig)
      var = tk.BooleanVar(wnd, value=True)
      self.var_fig_spectrum = var
      sm.add_checkbutton(label="Show Spectrum", variable=var, command=self._clt_fig)
      var = tk.BooleanVar(wnd, value=False)
      self.var_fig_power = var
      sm.add_checkbutton(label="Show Power", variable=var, command=self._clt_fig, state='disabled')
      var = tk.BooleanVar(wnd, value=False)
      self.var_fig_pitch = var
      sm.add_checkbutton(label="Show F0/Pitch", variable=var, command=self._clt_fig, state='disabled')
      sm.add_separator()
      sm.add_command(label="Open sound fork")
      menu.add_cascade(label="View", menu=sm)
      
      sm = tk.Menu(menu, tearoff=False)
      var = tk.BooleanVar(wnd, value=True)
      self.var_remove_dc_offset = var
      sm.add_checkbutton(label="Remove DC offset", variable=var)
      sm.add_checkbutton(label="Listen device loopback", command=self.audio_listen)
      sm.add_separator()
      sm.add_command(label="Advanced Options..", state='disabled')
      menu.add_cascade(label="Options", menu=sm)
      
      menu.add_command(label="Help", command=lambda: tkmsg.showinfo("Help", "I'm sorry but, this fucking world has no help :<"))

    # top: main panel
    frm11 = ttk.Frame(wnd)
    frm11.pack(side=tk.TOP, anchor=tk.N, fill=tk.BOTH, expand=tk.YES)
    if True:
      # left: reclist
      frm21 = ttk.Frame(frm11)
      frm21.pack(side=tk.LEFT, fill=tk.Y)
      if True:
        # up: current vocal symbol
        frm31 = ttk.Frame(frm21)
        frm31.pack(side=tk.TOP)
        if True:
          var = tk.StringVar(wnd, "")
          self.var_vsymtype = var
          lb = ttk.Label(frm31, textvariable=var, font=ft_label, background='#FFF')
          lb.pack(fill=tk.X, expand=tk.YES)
          self.lb_vsymtype = lb
        
        # down: vocal symbol-name list + vocal type list
        frm32 = ttk.Frame(frm21)
        frm32.pack(side=tk.BOTTOM, fill=tk.Y, expand=tk.YES)
        if True:
          frm41 = ttk.Frame(frm32)
          frm41.pack(side=tk.LEFT, fill=tk.Y)
          if True:
            sb = ttk.Scrollbar(frm41)
            sb.pack(side=tk.RIGHT, fill=tk.Y)

            var = tk.StringVar(wnd, "")
            self.var_vsyms_str = var
            ls = tk.Listbox(frm41, listvariable=var, width=4, font=ft_list, yscrollcommand=sb.set)
            sb.config(command=ls.yview)
            _ctl_rs = lambda evt: self._ctl_vsymtype_('symbol')
            ls.bind('<ButtonRelease-1>', _ctl_rs)
            ls.bind('<Return>', _ctl_rs)
            ls.bind('<Down>', _ctl_rs)
            ls.bind('<Up>', _ctl_rs)
            ls.pack(side=tk.LEFT, fill=tk.Y)
            self.ls_vsym = ls

          frm42 = ttk.Frame(frm32)
          frm42.pack(side=tk.RIGHT, fill=tk.Y)
          if True:
            sb = ttk.Scrollbar(frm42)
            sb.pack(side=tk.RIGHT, fill=tk.Y)

            var = tk.StringVar(wnd, "")
            self.var_vtypes_str = var
            ls = tk.Listbox(frm42, listvariable=var, width=4, font=ft_list, yscrollcommand=sb.set)
            sb.config(command=ls.yview)
            _ctl_rt = lambda evt: self._ctl_vsymtype_('type')
            ls.bind('<ButtonRelease-1>', _ctl_rt)
            ls.bind('<Return>', _ctl_rt)
            ls.bind('<Down>', _ctl_rt)
            ls.bind('<Up>', _ctl_rt)
            ls.pack(side=tk.LEFT, fill=tk.Y)
            self.ls_vtype = ls

      # right: graph/chart view
      frm22 = ttk.Frame(frm11)
      frm22.pack(side=tk.RIGHT, fill=tk.BOTH, expand=tk.YES)
      if True:
        fig = plt.figure(dpi=GRAPH_DPI)
        self.fig = fig
        cvs = FigureCanvasTkAgg(fig, frm22)
        self.cvs = cvs
        tkcvs = cvs.get_tk_widget()
        tkcvs.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)
        # tktb = NavigationToolbar2Tk(cvs, frm22)
        # tktb.update()
    
    # middle: save path
    frm12 = ttk.Frame(wnd)
    frm12.pack(fill=tk.X)
    if True:
      lb = ttk.Label(frm12, text="Save Path: ")
      lb.pack(side=tk.LEFT, anchor=tk.W)
      
      bt = ttk.Button(frm12, text="Select..", command=self._clt_savedir)
      bt.pack(side=tk.RIGHT, anchor=tk.E)

      var = tk.StringVar(wnd, value=os.path.abspath(SAVE_DIR))
      self.var_savedir = var
      et = ttk.Entry(frm12, textvariable=var, state='readonly')
      et.pack(fill=tk.X, padx=2, pady=2)
      
    # bottom: status bar
    frm13 = ttk.Frame(wnd)
    frm13.pack(side=tk.BOTTOM, anchor=tk.S, fill=tk.X)
    if True:
      var = tk.StringVar(wnd, "OK")
      self.var_stat_msg = var
      ttk.Label(frm13, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=tk.YES)
  
  def setup_workspace(self):
    self._clt_fig()

    self.list_load_('symbol', os.path.join(LIST_DIR, RECLIST_FILE))
    self.list_load_('type', os.path.join(LIST_DIR, TYPELIST_FILE))
  
  @property
  def cur_vsymtype(self):
    return self.cur_vsym + self.cur_vtype

  def alert(self, msg, level='info'):
    getattr(logger, level)(msg)
    self.var_stat_msg.set(msg)
    if level in ['error', 'fatal']:
      tkmsg.showerror("Error", msg)

  def _clt_fig(self):   # hide/show figs
    s_wv, s_sp, s_pw, s_pt = (self.var_fig_wave.get(), self.var_fig_spectrum.get(),
                              self.var_fig_power.get(), self.var_fig_pitch.get())
    nfigs = sum([s_wv, s_sp, s_pw, s_pt])
    fig = self.fig
    fig.clf()
    
    idx = 1
    if s_wv: self.fig_wave     = fig.add_subplot(nfigs, 1, idx, label='wave')     ; idx += 1
    else: self.fig_wave = None
    if s_sp: self.fig_spectrum = fig.add_subplot(nfigs, 1, idx, label='spectrum') ; idx += 1
    else: self.fig_spectrum = None
    if s_pw: self.fig_power    = fig.add_subplot(nfigs, 1, idx, label='power')    ; idx += 1
    else: self.fig_power = None
    if s_pt: self.fig_pitch    = fig.add_subplot(nfigs, 1, idx, label='pitch')
    else: self.fig_pitch = None
    self.fig_draw()     # redraw graphs

  def _clt_savedir(self):
    dp = tkfiledlg.askdirectory()
    if not dp: return
    dp = dp.replace('/', os.path.sep)
    self.var_savedir.set(dp)
  
  def _ctl_vsymtype_(self, what='symbol'):
    def _upd_vsym():
      idx = self.ls_vsym.curselection()
      idx = idx and idx or 0
      self.cur_vsym = self.ls_vsym.get(idx)
    def _upd_vtype():
      idx = self.ls_vtype.curselection()
      idx = idx and idx or 0
      self.cur_vtype = self.ls_vtype.get(idx)
    
    if what == 'symbol': _upd_vsym()
    elif what == 'type': _upd_vtype()
    elif what == 'all': _upd_vsym() ; _upd_vtype()
    if self.var_vsymtype.get() != self.cur_vsymtype:
      self.var_vsymtype.set(self.cur_vsymtype)
      self.fig_draw()
  
  def list_load_(self, what='symbol', fp=None):
    if not fp: fp = tkfiledlg.askopenfilename(initialdir=os.path.join(BASE_PATH, LIST_DIR),
                                              filetypes=[('txt', '*.txt'), ('All Files', '*')])
    if not fp: return
    
    with open(fp, 'rb') as fh: bdata = fh.read()
    data = fuck_encoding(bdata)
    if not data:
      self.alert("file %r decode error" % fp, 'error')
      return
    
    if what == 'symbol':
      self.var_vsyms_str.set(' '.join([t for t in data.split()]))
      self.ls_vsym.select_set(0)
    elif what == 'type':
      self.var_vtypes_str.set('"" ' + ' '.join([t for t in data.split()]))
      self.ls_vtype.select_set(0)
    self._ctl_vsymtype_()

  def audio_listen(self):
    def task_listen(working):
      nframes = max(1, int(FRAME_RATE / CHUNK * RECORD_SECONDS))
      while working.is_set():
        for i in range(nframes):
          chunks = self.stream.read(CHUNK)
          self.stream.write(chunks, CHUNK)
    
    if not self.working_listen.is_set():  # start listen
      self.working_listen.set()
      self.alert("Listening device loopback")
      self.thr_listen = Thread(target=task_listen, args=(self.working_listen,), daemon=True)
      self.thr_listen.start()
    else:                                 # stop listen
      self.working_listen.clear()
      self.thr_listen.join()
      self.thr_listen = None
      self.alert("Listening stopped")

  def audio_record(self):
    def task_record(working, rm_dc):
      chunks = [ ]
      nframes = max(1, int(FRAME_RATE / CHUNK * RECORD_SECONDS))
      while working.is_set():
        for i in range(nframes):
          chunks.append(self.stream.read(CHUNK))

      fp = os.path.join(TMP_DIR, self.cur_vsymtype + '.wav')
      with wave.open(fp, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(FRAME_RATE)
        wf.writeframes(b''.join(chunks))
      
      if rm_dc:
        seg = AudioSegment.from_wav(fp)
        seg.remove_dc_offset()
        seg.export(fp, format='wav')

      _, da_sp = wavfile.read(fp)
      # dt_rs, _ =  librosa.load(fp, sr=FRAME_RATE)
      self.records[self.cur_vsymtype] = (chunks, da_sp)

    if not self.working_record.is_set():  # start record
      self.working_record.set()
      self.alert('Recording %s' % self.cur_vsymtype)
      rm_dc = self.var_remove_dc_offset.get()
      self.thr_record = Thread(target=task_record, args=(self.working_record, rm_dc), daemon=True)
      self.thr_record.start()
    else:                                 # stop record
      self.working_record.clear()
      self.thr_record.join()
      self.thr_record = None
      self.alert('Recorded %s' % self.cur_vsymtype)
      self.fig_draw()

  def audio_play(self):
    chunks, _ = self.records[self.cur_vsymtype]
    if chunks: self.alert('Playing %s' % self.cur_vsymtype)
    for data in chunks: self.stream.write(data, CHUNK)

  def audio_save(self):
    src = os.path.join(BASE_PATH, TMP_DIR)
    dst = self.var_savedir.get()
    if not os.path.exists(dst): os.path.mkdir(dst)
    for fn in os.listdir(src):
      fp = os.path.join(src, fn)
      shutil.copy2(fp, dst)
    self.alert('records saved to %r' % dst)

  def fig_draw(self):
    if self.fig_wave:     self.fig_wave.clear()     ; self.fig_draw_wave()
    if self.fig_spectrum: self.fig_spectrum.clear() ; self.fig_draw_spectrum()
    if self.fig_power:    self.fig_power.clear()    ; self.fig_draw_power()
    if self.fig_pitch:    self.fig_pitch.clear()    ; self.fig_draw_pitch()
    self.cvs.draw()

  def fig_draw_wave(self):
    _, data = self.records[self.cur_vsymtype]
    if not len(data): return
    
    sample_interval = 1 / FRAME_RATE          # 采样点的时间间隔
    duration_time = len(data) / FRAME_RATE  # 声音信号的长度
    x = np.arange(0, duration_time, sample_interval)

    self.fig_wave.plot(x, data, 'blue')

  def fig_draw_wave2(self):
    _, data = self.records[self.cur_vsymtype]
    if not len(data): return

    rosadisplay.waveplot(data, sr=FRAME_RATE)
    plt.show()

  def fig_draw_spectrum(self):
    _, data = self.records[self.cur_vsymtype]
    if not len(data): return
    
    N = len(data)                       # 采样总帧数
    df = FRAME_RATE / (N - 1)           # 分辨率
    freq = [df*n for n in range(N)]     # 分辨率的倍数
    c = np.fft.fft(data) * 2 / N
    d = int(len(c)/2)                   # 频谱对称，故折半
    while freq[d] > 2048: d -= 10       # 低通2K
    
    self.fig_spectrum.xaxis.set_major_locator(plt.LogLocator(base=2))
    x, y = freq[:d-1], abs(c[:d-1])
    self.fig_spectrum.plot(x, y, 'r')

  def fig_draw_spectrum2(self):
    _, data = self.records[self.cur_vsymtype]
    if not len(data): return

    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    rosadisplay.specshow(Xdb, sr=FRAME_RATE, y_axis='log')

  def fig_draw_power(self):
    x = np.arange(0, 3, 0.01)
    y = np.sin(2 * np.pi * x)
    self.fig_power.plot(x, y)
  
  def fig_draw_pitch(self):
    x = np.arange(0, 3, 0.01)
    y = np.sin(2 * np.pi * x + 5)
    self.fig_pitch.plot(x, y)

# main
if __name__ == "__main__":
  # os.system('CHCP 65001 >NUL')
  os.chdir(BASE_PATH)
  if not os.path.exists(SAVE_DIR): os.mkdir(SAVE_DIR)
  if not os.path.exists(TMP_DIR): os.mkdir(TMP_DIR)

  App()
