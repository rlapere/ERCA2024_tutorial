def blockPrint():
    import sys,os
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    import sys,os
    sys.stdout = sys.__stdout__


def rescale(inxr,mn,mx,flip=False):
    import pandas as pd
    """
    function to rescale original data to the range of authorized pitch
    """
    
    mini = inxr.min()
    maxi = inxr.max()

    # normalize data to [0,1]
    inxr_ = (inxr-mini)/(maxi-mini)

    # apply a mapping to the range of authorized min/max pitch
    inxr_ = mn+inxr_*(mx-mn)

    # if you want to associate increasing temperature with lower pitch notes
    if flip==True:
        inxr_ = mn-inxr_+mx

    # make it an integer type because MIDI only handles semitones
    inxr_ = inxr_.astype(int)

    # store notes into a data set along with time steps
    df_ = pd.DataFrame({'val':inxr_,'step':range(len(inxr_))})
    
    return df_





def _to_chords_(df, key, minpitch, maxpitch):
    import numpy as np
    """
    function to map the original notes to the defined key
    """
    notes = range(len(np.arange(minpitch,maxpitch+1,1)))
    notes_ = np.arange(minpitch,maxpitch+1,1)
    dom = np.mod(notes,12)==key[0]
    tir = np.mod(notes,12)==key[1]
    qui = np.mod(notes,12)==key[2]
    sev = np.mod(notes,12)==key[3]
    auth = dom+tir+qui+sev
    auth_notes = notes_[auth]
    notin = df.val.values
    i=0
    for nn in notin:
        dist = np.abs(auth_notes-nn)
        tru_note = auth_notes[np.argmin(dist)]
        notin[i] = tru_note
        i=i+1
    df['val'] = notin
    return df




def extract_sdt(indata, kkeys, freq, minpitch, maxpitch):
    import numpy as np
    import pandas as pd
    """
    function to aggregate consecutive notes
    and includes info on duration/velocity
    """
    didif = [indata['step'].values[0]]
    indata = _to_chords_(indata, kkeys, minpitch, maxpitch)
    for i in np.arange(1,len(indata.val.values)):
        if indata['val'].values[i]==indata['val'].values[i-1]:
            didif = np.append(didif,indata['step'].values[i-1])
        else:
            didif = np.append(didif,indata['step'].values[i])
    indata['dif'] = didif
    steps = indata.groupby(['dif','val'],as_index=False).count().step.values
    vals = (indata.groupby(['dif','val'],as_index=False).mean().val.values).astype(int)
    new_df = pd.DataFrame({'note':vals,
                       'steps':np.cumsum(steps)-np.min(np.cumsum(steps)),
                       'duration':np.append(steps[1:],2),
                      'force':np.repeat(127,len(steps))})
    for j in np.arange(1,len(new_df.note.values)):
        if new_df.note.values[j] == new_df.note.values[j-1]:
            new_df.steps[j] = new_df.steps.values[j-1]
    dur = new_df.groupby(['steps'],as_index=False).sum().duration.values
    new_df = new_df.drop_duplicates(['steps','note'])
    new_df.duration = dur
    new_df['force'] = np.linspace(100,126,len(dur)).astype(int)
    new_df['steps'] = (new_df['steps'].values*freq).astype(int)
    new_df['duration'] = (new_df['duration'].values*freq).astype(int)
    new_df = new_df[['steps','note','force','duration']]
    return new_df




def apply_progression(indata,freq,freqbase,reperes,keys,chords,minpitch,maxpitch):
    import numpy as np
    import pandas as pd
    """
    apply the chord progression as defined by the variable chosen a bass line
    """    
    
    df_ = indata.copy()
    
    j = 0
    for r in reperes:
        keyloc = chords[str(int(keys[j]%12))]
        keyloc = np.append(keyloc[0],keyloc)
        if j <len(reperes)-1:
            dfloc = df_[(df_.step.values>=r) & (df_.step.values<reperes[j+1])]
            inloc = _to_chords_(dfloc, keyloc, minpitch, maxpitch)
            df_[(df_.step.values>=r) & (df_.step.values<reperes[j+1])] = inloc.astype(int)
        else:
            dfloc = df_[(df_.step.values>=r)]
            inloc = _to_chords_(dfloc, keyloc, minpitch, maxpitch)
            df_[df_.step>=r] = inloc.astype(int)
        j = j+1

    didif = [df_['step'].values[0]]
    for i in np.arange(1,len(df_.val.values)):
        if df_['val'].values[i]==df_['val'].values[i-1]:
            didif = np.append(didif,df_['step'].values[i-1])
        else:
            didif = np.append(didif,df_['step'].values[i])
    df_['dif'] = didif
    steps = df_.groupby(['dif','val'],as_index=False).count().step.values
    vals = (df_.groupby(['dif','val'],as_index=False).mean().val.values).astype(int)
    new_df = pd.DataFrame({'note':vals,
                       'steps':np.cumsum(steps)-np.min(np.cumsum(steps)),
                       'duration':np.append(steps[1:],2),
                      'force':np.repeat(127,len(steps))})
    for j in np.arange(1,len(new_df.note.values)):
        if new_df.note.values[j] == new_df.note.values[j-1]:
            new_df.steps[j] = new_df.steps.values[j-1]
    dur = new_df.groupby(['steps'],as_index=False).sum().duration.values
    new_df = new_df.drop_duplicates(['steps','note'])
    new_df.duration = dur
    new_df['force'] = np.linspace(100,126,len(dur)).astype(int)
    new_df['steps'] = (new_df['steps'].values*freq).astype(int)
    new_df['duration'] = (new_df['duration'].values*freq).astype(int)
    new_df = new_df[['steps','note','force','duration']]
    return new_df



def to_midi(infiles,nm):
    import numpy as np
    import pandas as pd
    from miditime.miditime import MIDITime
    """
    function to convert the data to MIDI file
    """
    blockPrint()
    mymidi = MIDITime(350, nm+'.mid') # 160 is the tempo
    for infile in infiles:
        music = np.array(pd.read_csv(infile,skiprows=1,header=None,index_col=0)).tolist()
        # Add a track with those notes
        mymidi.add_track(music)
    # Output the .mid file
    mymidi.save_midi()
    enablePrint()
    

def play_music(music_file):
    import pygame
    """
    stream music with mixer.music module in blocking manner
    this will stream the sound from disk while playing
    """
    clock = pygame.time.Clock()
    try:
        pygame.mixer.music.load(music_file)
        print("Music file %s is playing!" % music_file)
    except pygame.error:
        print("File %s not found! (%s)" % (music_file, pygame.get_error()))
        return
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        # check if playback has finished
        clock.tick(30)

        
def launch_music(music_file):
    import pygame
    freq = 44100    # audio CD quality
    bitsize = -16   # unsigned 16 bit
    channels = 2    # 1 is mono, 2 is stereo
    buffer = 1024    # number of samples
    pygame.mixer.init(freq, bitsize, channels, buffer)

    # optional volume 0 to 1.0
    pygame.mixer.music.set_volume(1.0)

    try:
        play_music(music_file)
    except KeyboardInterrupt:
        pygame.mixer.music.fadeout(1000)
        pygame.mixer.music.stop()
        raise SystemExit



    
def build_cmap(colors, N=100, reverse=''):
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    '''
    Build a custom colormap from a list of colors
    The colors are linearly interpolated
    The number of colors in the final colormap is N
    You can reverse the colormap by setting reverse='r'
    Originally created by Theodore Khadir
    ''' 
    if reverse == 'r':
        colors = [colors[len(colors)-1-i] for i in range(0, len(colors))]
        cmap = LinearSegmentedColormap.from_list('name', colors, N=N)
    else:
        cmap = LinearSegmentedColormap.from_list('name', colors, N=N)
    return cmap
