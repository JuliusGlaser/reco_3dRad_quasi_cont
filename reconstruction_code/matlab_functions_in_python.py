
import numpy as np
import numpy as np
from pathlib import Path

from pydicom.uid import generate_uid

import re
from typing import Any, Dict, List, Union


Number = Union[int, float]


def ls_comb(m, c, coil_axis=-1):
        numerator = np.sum(c.conj() * m, axis=coil_axis)
        denuminator = np.linalg.norm(c, axis=coil_axis)
        return np.divide(numerator, denuminator, where=(denuminator != 0))  # not standard division because of 0 elements

def _as_num(x: Any) -> Number:
    """
    MATLAB's cell2mat on a scalar cell is essentially "unwrap to scalar".
    Here we unwrap 1-element containers commonly seen in parsed data.
    """
    if isinstance(x, (list, tuple)) and len(x) == 1:
        return _as_num(x[0])
    return x

def _ensure_dict(d, key):
    """Ensure d[key] exists and is a dict; return it."""
    if key not in d or d[key] is None:
        d[key] = {}
    elif not isinstance(d[key], dict):
        raise TypeError(f"Expected params['{key}'] to be a dict, got {type(d[key])}")
    return d[key]


def _isfield(d, key):
    """MATLAB _isfield equivalent for dicts."""
    return isinstance(d, dict) and (key in d)

def cell2mat(x):
    if x is None:
        return np.array([])
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (list, tuple)):
        return np.array(x)
    return np.array([x])

def ncols(as_list):
    """MATLAB size(asList,2) best-effort."""
    if as_list is None:
        return 0
    if isinstance(as_list, np.ndarray):
        return as_list.shape[1] if as_list.ndim >= 2 else as_list.shape[0]
    if isinstance(as_list, (list, tuple)):
        if len(as_list) == 0:
            return 0
        first_row = as_list[0]
        if isinstance(first_row, (list, tuple, np.ndarray)):
            return len(first_row)
        return len(as_list)
    if isinstance(as_list, dict):
        return len(as_list)
    try:
        return len(as_list)
    except TypeError:
        return 0

def GetAcquisitionParams(params, mapped, importWipParameter=None):
    """
    MATLAB-like:
        params = struct()
        params = getAcquisitionParams(params)
    Python:
        params = {}
        params = getAcquisitionParams(params, mapped)
    """
    hdr = mapped[-1]['hdr']
    data = mapped[-1]['image']

    # ensure nested dict structure exists
    

    acq = params['acquisition']
    scn = params['scanner']
    pat = params['patient']
    stu = params['study']

    # -------------------------
    # GET ACQUISITION PARAMETERS
    # -------------------------
    acq['sequenceFilename'] = hdr['Config']['SequenceFileName']
    acq['sequenceDescription'] = hdr['Config']['SequenceDescription']
    acq['protocolName'] = hdr['Dicom']['tProtocolName']

    # averages/echos from your data dims
    eco_idx = data.dims.index('Eco')
    ave_idx = data.dims.index('Ave')
    acq['averages'] = int(data.shape[ave_idx])
    acq['echos'] = int(data.shape[eco_idx])

    # nucleus: MATLAB sRXSPEC.asNucleusInfo{1,1} and maybe {1,2}
    rx_info = hdr['MeasYaps']['sRXSPEC']['asNucleusInfo']
    nucleus = []

    # support either 2D (list-of-lists) or 1D list forms
    if isinstance(rx_info, list) and len(rx_info) > 0 and isinstance(rx_info[0], list):
        nucleus.append(rx_info[0][0].get('tNucleus', None))
        if len(rx_info[0]) > 1 and _isfield(rx_info[0][1], 'tNucleus'):
            nucleus.append(rx_info[0][1]['tNucleus'])
    else:
        nucleus.append(rx_info[0].get('tNucleus', None))
        if len(rx_info) > 1 and _isfield(rx_info[1], 'tNucleus'):
            nucleus.append(rx_info[1]['tNucleus'])
    acq['nucleus'] = nucleus

    acq['larmorConstant'] = hdr['Meas']['alLarmorConstant']

    # frequency: MATLAB sTXSPEC.asNucleusInfo{1} and maybe {2}
    tx_info = hdr['MeasYaps']['sTXSPEC']['asNucleusInfo']
    freq = []
    freq.append(tx_info[0]['lFrequency'])
    if len(tx_info) > 1 and _isfield(tx_info[1], 'lFrequency'):
        freq.append(tx_info[1]['lFrequency'])
    acq['frequency'] = freq

    acq['TA'] = hdr['MeasYaps']['lScanTimeSec']
    acq['TR'] = hdr['MeasYaps']['alTR'][0] / 1000.0

    if not _isfield(hdr['MeasYaps'], 'alTI'):
        acq['TI'] = []
    else:
        acq['TI'] = hdr['MeasYaps']['alTI'] / 1000.0

    TE_all = hdr['MeasYaps']['alTE']
    TE_all = [x / 1000.0 for x in TE_all]
    acq['TE'] = [float(TE_all[i]) for i in range(acq['echos'])]

    if not _isfield(hdr['Meas'], 'lRepetitions'):
        acq['repetitions'] = 1
    else:
        acq['repetitions'] = hdr['Meas']['lRepetitions']

    dwelltimeALL = cell2mat(hdr['MeasYaps']['sRXSPEC']['alDwellTime'])
    dwelltime = [None, None]
    dwelltime[0] = float(dwelltimeALL[0]) / 1000.0
    if (len(dwelltimeALL) > 1) and (dwelltimeALL[0] != dwelltimeALL[1]):
        dwelltime[1] = float(dwelltimeALL[1]) / 1000.0
    else:
        dwelltime[1] = float(dwelltimeALL[0]) / 1000.0
    acq['dwelltime'] = dwelltime

    acq['radialSamples'] = hdr['MeasYaps']['sKSpace']['lBaseResolution']

    # pulseAmplitude
    pulseAmp = []
    if not _isfield(tx_info[0], 'flReferenceAmplitude'):
        pulseAmp.append(0)
    else:
        pulseAmp.append(tx_info[0]['flReferenceAmplitude'])
        if len(tx_info) > 1 and _isfield(tx_info[1], 'flReferenceAmplitude'):
            pulseAmp.append(tx_info[1]['flReferenceAmplitude'])
    acq['pulseAmplitude'] = pulseAmp

    # flipAngle
    flipAngleDegree = hdr['MeasYaps'].get('adFlipAngleDegree', None)
    if (not _isfield(hdr['MeasYaps'], 'adFlipAngleDegree')) or (flipAngleDegree is None) or (len(flipAngleDegree) == 0):
        acq['flipAngle'] = 0
    else:
        fa = [flipAngleDegree[0]]
        if len(flipAngleDegree) > 1:
            fa.append(flipAngleDegree[1])
        acq['flipAngle'] = fa

    # channels: MATLAB size(...,2) and optional second
    coil = hdr['MeasYaps']['sCoilSelectMeas']['aRxCoilSelectData']
    # channels: size(...,2) -> number of columns. 
    # # Here I assume asList is a 2D array/list-of-lists where columns = len(asList[0]). 
    channels = [] 
    # asList1 = hdr['MeasYaps']['sCoilSelectMeas']['aRxCoilSelectData'][0]['asList'][0] 
    # first_key = next(iter(asList1)) # "a" 
    # first_value = asList1[first_key] # 10 
    # channels.append(first_value) #FIXME: Not sure if that works for other datasets 
    # #FIXME: channels[1] = ????

    acq['channels'] = len(hdr['MeasYaps']['sCoilSelectMeas']['aRxCoilSelectData'][0]['asList'])

    # -------------------------
    # GET WIP PARAMETERS
    # -------------------------
    WipMemBlock = hdr['MeasYaps']['sWipMemBlock']
    params = import_wip_parameter(WipMemBlock, params)

    # -------------------------
    # GET SCANNER PARAMETERS
    # -------------------------
    scn['institutionAdress'] = hdr['Dicom']['InstitutionAddress']
    scn['institutionName'] = hdr['Dicom']['InstitutionName']
    scn['manufacturer'] = hdr['Dicom']['Manufacturer']
    scn['modality'] = hdr['Dicom']['Modality']
    scn['model'] = hdr['Dicom']['ManufacturersModelName']
    scn['fieldstrength'] = hdr['Dicom']['flMagneticFieldStrength']
    scn['deviceSerialNumber'] = str(hdr['Dicom']['DeviceSerialNumber'])

    softwareVersion_raw = hdr['Dicom']['SoftwareVersions']
    scn['softwareVersion'] = "".join(str(softwareVersion_raw).split())
    scn['coilName'] = hdr['Dicom']['TransmittingCoil']

    # -------------------------
    # GET PATIENT AND STUDY PARAMETERS
    # -------------------------
    pat['patientID'] = hdr['Config']['PatientID']
    pat['patientName'] = hdr['Config']['PatientName']

    patientSex_code = hdr['Config']['PatientSex']
    if patientSex_code == '1':
        pat['patientSex'] = 'F'
    elif patientSex_code == '2':
        pat['patientSex'] = 'M'
    else:
        pat['patientSex'] = 'O'

    pat['patientBirthday'] = hdr['Config']['PatientBirthDay']
    pat['patientAge'] = hdr['Dicom']['flPatientAge']
    pat['patientWeight'] = hdr['Dicom']['flUsedPatientWeight']
    pat['patientPosition'] = hdr['Config']['PatientPosition']

    stu['studyDescription'] = hdr['Config']['tStudyDescription']
    stu['dicomUID'] = generate_uid()

    return params

def SetReconParams(config, params, mapped):

    hdr = mapped[-1]['hdr']

    config_reco = config['recon_parameter']
    reco = params['reconstruction']

    reco['kvecMethod'] = config_reco['kvecMethod']  # 1= theoretical, 2 = measured

    FOV = config_reco['FOV']
    reco['size'] = {}
    reco['size']['fov'] = [FOV] * 3

    SizeFilling = FOV 
    reco['sizeFilling'] = [SizeFilling] * 3

    if config_reco['translation_adjustment_vol']:
        trans_adj_vol = hdr['Phoenix']['sAdjData']['sAdjVolume']['sPosition']
        shift_sag = trans_adj_vol.get('dSag', 0.0) / 10.0
        shift_cor = trans_adj_vol.get('dCor', 0.0) / 10.0
        shift_tra = trans_adj_vol.get('dTra', 0.0) / 10.0
    else:
        shift_sag = config_reco['translation_sag']
        shift_cor = config_reco['translation_cor']
        shift_tra = config_reco['translation_tra']

    reco['translation'] = [shift_cor, shift_sag, shift_tra]
    reco['rotation'] = [0.0, 0.0, 0.0]

    params['goldentest'] = 1

    reco['filter'] = {}
    reco['filter']['type'] = config_reco['filter_type']
    reco['filter']['sigma'] = config_reco['filter_sigma']

    reco['gridOS'] = config_reco['gridOS']
    reco['kbWidth'] = config_reco['kbWidth']

    reco['b0correction'] = config_reco['b0correction'] # 2 = no B0 correction, 1 = with B0 correction

    reco['noReplicas'] = 1

    if params['goldentest'] is None:            #FIXME: Don't know if relevant
        reco['selectiveProjections'] = [0, 2000]


    # --- CHOOSE RECONSTRUCTION PARAMETERS (OUTPUT)
    out = _ensure_dict(params, "output")
    out["nifti"] = 0
    out["dicom"] = 0
    out["outputMag"] = 1
    out["outputPhase"] = 0
    out["outputComp"] = 0

    # --- CHOOSE RECONSTRUCTION PARAMETERS (MULTICHANNEL)
    # channels 1..32 in MATLAB
    reco["selectiveChannels"] = np.arange(1, 33, dtype=int)
    reco["correctUSM"] = 0  # 1 = correct image with universal sensitivity map

    # Acquisition sub-struct requirements:
    # params.acquisition.channels (array-like)
    # params.acquisition.echos (int)
    # params.acquisition.averages (int)
    # params.acquisition.goldenAngle (bool)
    # params.acquisition.projections (int)
    # params.acquisition.TRFrames (array-like, index 0 used)
    acq = _ensure_dict(params, "acquisition")

    # multichannel settings
    max_channels = acq['channels']
    if max_channels > 1:
        multich = _ensure_dict(reco, "multichannel")
        multich["typeCombination"] = config_reco['multichannel_combination'] # 1=ADC, otherwise SOS #TODO: config parameter

        if multich["typeCombination"] == 1:
            adc = _ensure_dict(multich, "adc")
            adc["blockSize"] = [10, 10, 10]
            adc["interpFactor"] = [2, 2, 2]
            adc["overlap"] = [8, 8, 8]

    # --- PROCESS CHOSEN RECONSTRUCTION PARAMETERS
    # sizeFilling(i) = max(sizeFilling(i), size.fov(i))
    fov = np.array(reco['size']["fov"], dtype=float)
    size_filling = np.array(reco["sizeFilling"], dtype=float)
    reco["sizeFilling"] = np.maximum(size_filling, fov).astype(int).tolist()

    # selectiveEchos <= acquisition.echos and ~= 0
    acq_echos = int(acq.get("echos", 0))
    echos = list((np.arange(acq_echos, dtype=int)+1).squeeze())     #FIXME: Don't know what here should happen
    reco["selectiveEchos"] = echos

    # selectiveAverages <= acquisition.averages and ~= 0
    acq_avgs = int(acq.get("averages", 0))
    avgs = (np.arange(acq_avgs, dtype=int)+1).squeeze() # FIXME: Don't know what here should happen
    reco["selectiveAverages"] = avgs

    # selectiveProjections logic
    # FIXME: this part is unclear; leaving as default for now
    # golden = bool(acq.get("goldenAngle", False))
    # if not golden:
    #     # MATLAB else branch sets:
    #     # selectiveProjections(1,1) = 1
    #     # selectiveProjections(1,2) = projections * TRFrames(1)
    #     projections = int(acq.get("projections", 0))
    #     tr_frames = acq.get("TRFrames", [0])
    #     tr1 = int(tr_frames[0]) if len(tr_frames) > 0 else 0
    #     reco["selectiveProjections"] = [1, projections * tr1]
    # else:
    #     # In MATLAB the golden-angle branch mostly has commented-out clamping.
    #     # We leave reco["selectiveProjections"] as-is if already set,
    #     # otherwise you may want a sensible default:
    #     if "selectiveProjections" not in reco:
    #         reco["selectiveProjections"] = np.array([[1, 2000]], dtype=int)

    # selectiveChannels <= max(acquisition.channels)
    sel_ch = np.array(params["acquisition"]["channels"], dtype=int)
    reco["selectiveChannels"] = sel_ch

    # --- DETERMINE BETA FOR KAISER BESSEL GRIDDING
    grid_os = float(reco.get("gridOS", 2))
    kb_width = float(reco.get("kbWidth", 4.0))

    def _default_beta(kbw, gos):
        return np.pi * np.sqrt((kbw / gos * (gos - 0.5)) ** 2 - 0.8)

    if grid_os == 2:
        mapping = {
            1.5: 6.6875,
            2.0: 9.1375,
            2.5: 11.5250,
            3.0: 13.9086,
            3.5: 16.2734,
            4.0: 18.5547,
        }
        beta = mapping.get(kb_width, _default_beta(kb_width, grid_os))
    elif grid_os == 1:
        mapping = {
            1.5: 1.9980,
            2.0: 2.3934,
            2.5: 3.3800,
            3.0: 4.2054,
            3.5: 4.9107,
            4.0: 5.7567,
            4.5: 6.6291,
            5.0: 7.4302,
        }
        beta = mapping.get(kb_width, _default_beta(kb_width, grid_os))
    else:
        beta = _default_beta(kb_width, grid_os)

    reco["kbBeta"] = float(beta)

    # --- GET NOISEFILE PARAMETERS FOR ADC (placeholder)
    # MATLAB portion depends on:
    # - file dialogs (uigetfile)
    # - Siemens twix reader mapVBVD
    # - and substantial numeric processing
    #
    # In Python you’ll likely plug in your own "load_noise_twix(...)" function and compute
    # covariance/correlation matrices.
    if (max_channels > 1) and (reco.get("multichannel", {}).get("typeCombination", None) == 1):
        adc = _ensure_dict(reco["multichannel"], "adc")

        # If you want to mimic MATLAB behavior:
        # - if no patientFolder: ask user to pick file (GUI)
        # - else use noisefile='na_noise_rawdata' in patientFolder
        #
        # Here we just prepare slots; you fill them after you implement your IO.
        if not _isfield(params, "patientFolder"):
            adc.setdefault("covarianceMatrix_inv", np.eye(max_channels))
        else:
            # TODO: load noise data and compute covariance/correlation matrices
            adc.setdefault("covarianceMatrix_inv", np.eye(max_channels))

    # --- LOAD UNIVERSAL SENSITIVITY MAP FOR INTENSITY CORRECTION (placeholder)
    # MATLAB loads .mat, does FFTs, cut edges, etc.
    # If you want, we can port this too once you confirm:
    # - file path handling
    # - what CutEdges3D should do in Python
    # - what array layout you want (numpy fft conventions)
    if (reco.get("correctUSM", 0) == 1) and (max_channels > 1):
        # TODO: implement USM loading + FFT pipeline
        # reco["senseMap"] = sense_map
        pass

    return params

def createParams(config, mapped):
    params = {}

    params.setdefault('acquisition', {})
    params.setdefault('scanner', {})
    params.setdefault('patient', {})
    params.setdefault('study', {})
    params.setdefault('reconstruction', {})
    params.setdefault('path', Path(config['path']).as_posix())
    params.setdefault('filename', config['file'])
    params.setdefault('offresonant', {})
    params.setdefault('goldentest', None)
    params.setdefault('output', {})
    params.setdefault('offresonant', {})

    params = GetAcquisitionParams(params, mapped)
    params = SetReconParams(config, params, mapped)

    return params

def PrepRaw(params, mapped):
    acq = params['acquisition']
    if acq['nucleus'][0] == '23Na':
        pass  # TODO: implement sodium preprocessing if needed
    else:
        data = mapped[-1]['image'][:].squeeze()     # Echos x Projections x Channels x RadialSamples
        if acq['channels'] == 1:
            data = data[:,np.newaxis,...]   
            data = data.transpose(0, 2, 1, 3)       # Echos x Channels x Projections x RadialSamples
        else:
            data = data.transpose(0, 2, 1, 3)       # Echos x Channels x Projections x RadialSamples

    combined_rawdata = np.zeros((acq['channels'], acq['projections'] * acq['echos'], acq['radialSamples']), dtype=np.complex64)

    for i in range(acq['echos']):
        temp_rawdata = data[i, ...]
        start_pos = i * acq['projections']
        combined_rawdata[:, start_pos:start_pos + acq['projections'], :] = temp_rawdata

    return combined_rawdata

def import_wip_parameter(WipMemBlock: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Python version of importWipParameter(WipMemBlock, paramsAcq)

    Notes:
    - Expects params['acquisition'] to be a dict-like object.
    - Expects WipMemBlock.alFree to be a 1-based MATLAB-like list in meaning.
      We implement indexing with Python 0-based by subtracting 1.
    """
    paramsAcq: Dict[str, Any] = params["acquisition"]

    sequenceIdentifier = paramsAcq.get("sequenceFilename", "")

    # Convenience accessor for MATLAB 1-based indexing: alFree(i) -> alFree[i-1]
    alFree= WipMemBlock['alFree']

    def af(i: int) -> Number:
        return _as_num(alFree[i - 1])

    if re.search(r"LG_DA_3D_RAD_ME", sequenceIdentifier):
        # SEQUENCE LG_DA_3D_RAD_ME
        if len(alFree) == 11:
            paramsAcq["gradAmp"] = af(9) / 100
            paramsAcq["t0"] = af(8) / 1000
            paramsAcq["pd"] = af(7)
            paramsAcq["slewrate"] = af(10)
            paramsAcq["projections"] = af(11)
            paramsAcq["TRFrames"] = [1, 1]
            paramsAcq["projectionsFrame"] = [x * paramsAcq["projections"] for x in paramsAcq["TRFrames"]]
        else:
            paramsAcq["gradAmp"] = af(10) / 100
            paramsAcq["t0"] = af(6) / 1000
            paramsAcq["pd"] = af(5)
            paramsAcq["slewrate"] = af(11)
            paramsAcq["projections"] = af(12)
            paramsAcq["TRFrames"] = [1, 1]
            paramsAcq["projectionsFrame"] = [x * paramsAcq["projections"] for x in paramsAcq["TRFrames"]]

        paramsAcq["samplingMode"] = "Radial"
        paramsAcq["mrAquisitionType"] = "3D"
        paramsAcq["coordinatesVersion"] = 2

        echos = int(paramsAcq.get("echos", 0))
        paramsAcq["echoStruct"] = [1] * echos

        paramsAcq["goldenAngle"] = 1 if af(8) == 3 else 0

    elif re.search(r"PJ_DA_3D_RAD", sequenceIdentifier):
        paramsAcq["gradAmp"] = [af(36) / 100]
        paramsAcq["t0"] = [af(31) / 1000]
        paramsAcq["pd"] = [af(26)]
        paramsAcq["slewrate"] = [af(41)]

        paramsAcq["projections"] = af(21) * 12
        paramsAcq["goldenAngle"] = bool(_as_num(alFree[0]))  # ~isempty(cell2mat(alFree(1)))

        paramsAcq["samplingMode"] = "Radial"
        paramsAcq["mrAquisitionType"] = "3D"
        paramsAcq["coordinatesVersion"] = 1

        echos = int(paramsAcq.get("echos", 0))
        paramsAcq["echoStruct"] = [1] * echos

        paramsAcq["TRFrames"] = [1, 1]
        paramsAcq["projectionsFrame"] = [x * paramsAcq["projections"] for x in paramsAcq["TRFrames"]]

    elif re.search(r"MR_DA_3D_RAD", sequenceIdentifier):
        paramsAcq["gradAmp"] = [af(36) / 100]  # %31
        paramsAcq["t0"] = [af(31) / 1000]      # %26
        paramsAcq["pd"] = [af(26)]             # %21
        paramsAcq["slewrate"] = [af(41)]       # %32

        paramsAcq["projections"] = af(21)      # %16

        paramsAcq["TEspacing"] = af(51)        # %41
        paramsAcq["TRFrames"] = [1, 1]
        paramsAcq["projectionsFrame"] = [x * paramsAcq["projections"] for x in paramsAcq["TRFrames"]]

        paramsAcq["goldenAngle"] = bool(_as_num(alFree[0]))

        paramsAcq["samplingMode"] = "Radial"
        paramsAcq["mrAquisitionType"] = "3D"
        paramsAcq["coordinatesVersion"] = 1

        echos = int(paramsAcq.get("echos", 0))
        paramsAcq["echoStruct"] = [1] * echos

        paramsAcq["echospacing"] = af(51)

    elif re.search(r"TW_DA_3D_RAD_IL", sequenceIdentifier):
        # SEQUENCE TW_DA_3D_RAD_basis
        paramsAcq["gradAmp"] = [af(36) / 100, af(37) / 100]
        paramsAcq["t0"] = [af(31) / 1000, af(32) / 1000]
        paramsAcq["pd"] = [af(26), af(27)]
        paramsAcq["slewrate"] = [af(41), af(42)]
        paramsAcq["projections"] = af(21)
        paramsAcq["TRFrames"] = [af(46), af(47)]
        paramsAcq["projectionsFrame"] = [x * paramsAcq["projections"] for x in paramsAcq["TRFrames"]]

        paramsAcq["goldenAngle"] = bool(_as_num(alFree[0]))

        paramsAcq["samplingMode"] = "Radial"
        paramsAcq["mrAquisitionType"] = "3D"
        paramsAcq["coordinatesVersion"] = 1

        if re.search(r"TW_DA_3D_RAD_IL_basisNew", sequenceIdentifier):
            paramsAcq["acquisitionMode"] = 1
        else:
            paramsAcq["acquisitionMode"] = af(5)

        if paramsAcq["acquisitionMode"] == 1:
            paramsAcq["echoStruct"] = [1, 2]
        elif paramsAcq["acquisitionMode"] == 2:
            paramsAcq["echoStruct"] = [1, 2, 1, 2]

    elif re.search(r"TW_DA_3D_RAD_basis", sequenceIdentifier):
        # SEQUENCE TW_DA_3D_RAD_basis
        paramsAcq["gradAmp"] = [af(36) / 100, af(37) / 100]
        paramsAcq["t0"] = [af(31) / 1000, af(32) / 1000]
        paramsAcq["pd"] = [af(26), af(27)]
        paramsAcq["slewrate"] = [af(41), af(42)]
        paramsAcq["projections"] = af(21)
        paramsAcq["TRFrames"] = [af(46), af(47)]
        paramsAcq["projectionsFrame"] = [x * paramsAcq["projections"] for x in paramsAcq["TRFrames"]]

        paramsAcq["goldenAngle"] = bool(_as_num(alFree[0]))

        paramsAcq["samplingMode"] = "Radial"
        paramsAcq["mrAquisitionType"] = "3D"
        paramsAcq["coordinatesVersion"] = 1

        paramsAcq["acquisitionMode"] = af(5)

        echos = int(paramsAcq.get("echos", 0))
        if paramsAcq["acquisitionMode"] == 1:
            paramsAcq["echoStruct"] = [1] * echos
        elif paramsAcq["acquisitionMode"] in (2, 3):
            paramsAcq["echoStruct"] = [1, 2]
        elif paramsAcq["acquisitionMode"] == 4:
            paramsAcq["echoStruct"] = [1, 2, 1, 2]

    elif re.search(r"TW_DA_3D_RAD_ME", sequenceIdentifier):
        # SEQUENCE TW_DA_3D_RAD_ME
        paramsAcq["gradAmp"] = [af(36) / 100, af(37) / 100]
        paramsAcq["t0"] = [af(31) / 1000, af(32) / 1000]
        paramsAcq["pd"] = [af(26)]
        paramsAcq["slewrate"] = [af(41), af(42)]
        paramsAcq["projections"] = af(21)
        paramsAcq["TRFrames"] = [1, 1]
        paramsAcq["projectionsFrame"] = [x * paramsAcq["projections"] for x in paramsAcq["TRFrames"]]

        paramsAcq["goldenAngle"] = bool(_as_num(alFree[0]))

        paramsAcq["samplingMode"] = "Radial"
        paramsAcq["mrAquisitionType"] = "3D"
        paramsAcq["coordinatesVersion"] = 1

        paramsAcq["acquisitionMode"] = af(5)

        echos = int(paramsAcq.get("echos", 0))
        if paramsAcq["acquisitionMode"] == 1:
            paramsAcq["echoStruct"] = [1] * echos
        elif paramsAcq["acquisitionMode"] == 2:
            paramsAcq["echoStruct"] = [1, 2, 3, 4]
            # channels(2) = channels(1)
            ch = paramsAcq.get("channels")
            if isinstance(ch, list) and len(ch) >= 1:
                if len(ch) < 2:
                    ch.append(ch[0])
                else:
                    ch[1] = ch[0]
                paramsAcq["channels"] = ch

    elif re.search(r"TW_DA_3D_RAD_DQF", sequenceIdentifier):
        # SEQUENCE TW_DA_3D_RAD_DQF
        paramsAcq["gradAmp"] = [af(36) / 100]
        paramsAcq["t0"] = [af(31) / 1000]
        paramsAcq["pd"] = [af(26)]
        paramsAcq["slewrate"] = [af(41)]
        paramsAcq["projections"] = af(21)
        paramsAcq["TRFrames"] = [1, 1]
        paramsAcq["projectionsFrame"] = [x * paramsAcq["projections"] for x in paramsAcq["TRFrames"]]

        paramsAcq["goldenAngle"] = bool(_as_num(alFree[0]))

        paramsAcq["samplingMode"] = "Radial"
        paramsAcq["mrAquisitionType"] = "3D"
        paramsAcq["coordinatesVersion"] = 1

        paramsAcq["acquisitionMode"] = af(5)

        echos = int(paramsAcq.get("echos", 0))
        paramsAcq["echoStruct"] = [1] * echos

    elif re.search(r"TW_DA_3D_RAD_IRcombined", sequenceIdentifier):
        # SEQUENCE TW_DA_3D_RAD_IRcombined
        paramsAcq["gradAmp"] = [af(36) / 100, af(37) / 100]
        paramsAcq["t0"] = [af(31) / 1000, af(32) / 1000]
        paramsAcq["pd"] = [af(26), af(27)]
        paramsAcq["slewrate"] = [af(41), af(42)]
        paramsAcq["projections"] = af(21)
        paramsAcq["TRFrames"] = [1, 1]
        paramsAcq["projectionsFrame"] = [x * paramsAcq["projections"] for x in paramsAcq["TRFrames"]]

        paramsAcq["goldenAngle"] = bool(_as_num(alFree[0]))

        paramsAcq["samplingMode"] = "Radial"
        paramsAcq["mrAquisitionType"] = "3D"
        paramsAcq["coordinatesVersion"] = 2

        echos = int(paramsAcq.get("echos", 0))
        paramsAcq["echoStruct"] = list(range(1, echos + 1))  # 1:echos

    elif re.search(r"TW_DA_3D_RAD_simple", sequenceIdentifier):
        # SEQUENCE TW_DA_3D_RAD_simple
        paramsAcq["gradAmp"] = [af(36) / 100, af(37) / 100]
        paramsAcq["t0"] = [af(31) / 1000, af(32) / 1000]
        paramsAcq["pd"] = [af(26), af(27)]
        paramsAcq["slewrate"] = [af(41), af(42)]
        paramsAcq["projections"] = af(21)
        paramsAcq["TRFrames"] = [1, 1]
        paramsAcq["projectionsFrame"] = [x * paramsAcq["projections"] for x in paramsAcq["TRFrames"]]

        paramsAcq["goldenAngle"] = bool(_as_num(alFree[0]))

        paramsAcq["samplingMode"] = "Radial"
        paramsAcq["mrAquisitionType"] = "3D"
        paramsAcq["coordinatesVersion"] = 1

        # channels(2) = channels(1)
        ch = paramsAcq.get("channels")
        if isinstance(ch, list) and len(ch) >= 1:
            if len(ch) < 2:
                ch.append(ch[0])
            else:
                ch[1] = ch[0]
            paramsAcq["channels"] = ch

        paramsAcq["acquisitionMode"] = 1
        if paramsAcq["acquisitionMode"] == 1:
            paramsAcq["echoStruct"] = [1, 2]
        elif paramsAcq["acquisitionMode"] == 2:
            paramsAcq["echoStruct"] = [1, 2, 1, 2]

    elif re.search(r"MM_DA_3D_RAD", sequenceIdentifier):
        # SEQUENCE MM_DA_3D_RAD
        paramsAcq["gradAmp"] = af(7) / 100
        paramsAcq["t0"] = af(8) / 1000
        paramsAcq["pd"] = af(1)
        paramsAcq["slewrate"] = af(6)
        paramsAcq["projections"] = af(2)
        paramsAcq["TRFrames"] = [1, 1]
        paramsAcq["projectionsFrame"] = [x * paramsAcq["projections"] for x in paramsAcq["TRFrames"]]

        paramsAcq["samplingMode"] = "Radial"
        paramsAcq["mrAquisitionType"] = "3D"
        paramsAcq["coordinatesVersion"] = 1

        echos = int(paramsAcq.get("echos", 0))
        paramsAcq["echoStruct"] = [1] * echos

        paramsAcq["goldenAngle"] = 1 if af(3) == 8 else 0

    elif re.search(r"NE_DA_3D_RAD_AFI", sequenceIdentifier):
        # SEQUENCE NE_DA_3D_RAD_AFI
        paramsAcq["gradAmp"] = af(10) / 100
        paramsAcq["t0"] = af(11) / 1000
        paramsAcq["pd"] = af(1)
        paramsAcq["slewrate"] = af(9)
        paramsAcq["projections"] = af(3)
        paramsAcq["TRFrames"] = [1, 1]
        paramsAcq["projectionsFrame"] = [x * paramsAcq["projections"] for x in paramsAcq["TRFrames"]]

        paramsAcq["samplingMode"] = "Radial"
        paramsAcq["mrAquisitionType"] = "3D"
        paramsAcq["coordinatesVersion"] = 1

        paramsAcq["echos"] = 2
        paramsAcq["echoStruct"] = [1] * int(paramsAcq["echos"])

        paramsAcq["goldenAngle"] = 1 if af(3) == 8 else 0

    elif re.search(r"B1_mapping", sequenceIdentifier):
        # SEQUENCE B1_mapping
        paramsAcq["gradAmp"] = af(11) / 100
        paramsAcq["t0"] = af(10) / 1000
        paramsAcq["pd"] = af(9)
        paramsAcq["slewrate"] = af(12)
        paramsAcq["projections"] = af(13)
        paramsAcq["TRFrames"] = [1, 1]
        paramsAcq["projectionsFrame"] = [x * paramsAcq["projections"] for x in paramsAcq["TRFrames"]]

        paramsAcq["samplingMode"] = "Radial"
        paramsAcq["mrAquisitionType"] = "3D"
        paramsAcq["coordinatesVersion"] = 2

        echos = int(paramsAcq.get("echos", 0))
        paramsAcq["echoStruct"] = [1] * echos

        paramsAcq["goldenAngle"] = 0
        print("TODO: GoldenAngle")

    else:
        print(
            "WIP Parameters were not read successfully! "
            "This may cause other errors in the reconstruction process!"
        )

    return params

def get_quasi_continuous_params(params: Any) -> tuple[int, np.ndarray]:
    """
    Port of:
      [steps, meanTimes] = getQuasiContinuousParams(params)

    Assumptions (matching the MATLAB code):
    - params.acquisition.TEspacing is in microseconds [µs]
    - params.acquisition.TE is in milliseconds [ms] (since MATLAB multiplies by 1e3 -> µs)
    - params.acquisition.projections is per-echo projections, multiplied by numCon (echos)
    - params.slidingwindowpros, params.slidingwindowstep are integers in "# projections"
    """
    acq = params["acquisition"] if isinstance(params, dict) else params.acquisition

    echospacing = acq["TEspacing"] if isinstance(acq, dict) else acq.TEspacing  # [µs]
    numCon = acq["echos"] if isinstance(acq, dict) else acq.echos
    projections = acq["projections"] if isinstance(acq, dict) else acq.projections
    TE = acq["TE"] if isinstance(acq, dict) else acq.TE

    slidingwindowpros = params["slidingwindowpros"] if isinstance(params, dict) else params.slidingwindowpros
    slidingwindowstep = params["slidingwindowstep"] if isinstance(params, dict) else params.slidingwindowstep

    pros = projections * numCon  # [#]
    steps = int((pros - slidingwindowpros) / slidingwindowstep + 1)

    echostep = echospacing / pros * numCon

    GRAD_RASTER_TIME = 10  # [µs]

    # MATLAB: (0:pros-1) * echostep + params.acquisition.TE(1)*1e3;
    # TE(1) in MATLAB -> TE[0] in Python
    T_raw = np.arange(pros, dtype=float) * echostep + float(TE[0]) * 1e3

    # MATLAB: ceil(floor(T_raw) / GRAD_RASTER_TIME) * GRAD_RASTER_TIME;
    T_rounded = np.ceil(np.floor(T_raw) / GRAD_RASTER_TIME) * GRAD_RASTER_TIME

    # MATLAB: floor((length(T_rounded) - slidingwindowpros)/slidingwindowstep) + 1;
    numBins = int(np.floor((T_rounded.size - slidingwindowpros) / slidingwindowstep) + 1)

    meanTimes = np.zeros(numBins, dtype=float)
    T_rounded_binned = []
    for i in range(numBins):
        # MATLAB is 1-based:
        # startIdx = (i - 1) * slidingwindowstep + 1;
        # endIdx = startIdx + slidingwindowpros - 1;
        # mean(T_rounded(startIdx:endIdx))
        start = i * slidingwindowstep               # 0-based inclusive
        end = start + slidingwindowpros             # 0-based exclusive
        meanTimes[i] = float(np.mean(T_rounded[start:end]))
        T_rounded_binned.append(T_rounded[start:end])
    return steps, meanTimes, np.array(T_rounded_binned)

def getImageArrays(params: Any, steps: int) -> tuple[np.ndarray, np.ndarray]:

    sizeFilling = params["reconstruction"]["sizeFilling"]
    nCh = params["acquisition"]["channels"]
    nShifts = len(params["offresonant"]["shift"]) 

    finalImages_array = np.zeros((nShifts, steps, 1, 1, sizeFilling[1], sizeFilling[0]), dtype=np.complex64)

    if nCh > 1:
        images_array = np.zeros((sizeFilling[0], sizeFilling[1], 1, nCh, steps, nShifts), dtype=np.complex64)
    else:
        images_array = np.array([])

    return finalImages_array, images_array

def getProjectionAngles(params: Any, ech: int) -> tuple[np.ndarray, np.ndarray]:
    acq = params["acquisition"] if isinstance(params, dict) else params.acquisition
    reco = params["reconstruction"] if isinstance(params, dict) else params.reconstruction

    numProjections =  reco["selectiveProjections"][1] - reco["selectiveProjections"][0]  # MATLAB 1-based indexing

    if acq["goldenAngle"]:
        lambda1 = 0.465571231876768
        lambda2 = 0.682327803828019
        dPolarAngle = np.zeros(numProjections, dtype=float)
        dAzimutalAngle = np.zeros(numProjections, dtype=float)
        for lk in range(reco["selectiveProjections"][0], reco["selectiveProjections"][1]):  # MATLAB 1-based indexing
            dH = -1.0 + (2.0 * (lk) * lambda1) % 2.0
            dPolarAngle[lk-reco["selectiveProjections"][0]] = np.arccos(dH)
            dAzimutalAngle[lk-reco["selectiveProjections"][0]] = (lk) * lambda2 * 2.0 * np.pi % (2.0 * np.pi)
    else:
        dPolarAngle = np.zeros(numProjections, dtype=float)
        dAzimutalAngle = np.zeros(numProjections, dtype=float)
        for lk in range(reco["selectiveProjections"][0], reco["selectiveProjections"][1]):  # MATLAB 1-based indexing
            dH = -1 + 2 * (lk) / (numProjections - 1)
            dPolarAngle[lk-reco['selectiveProjections'][0]] = np.arccos(dH)
            if lk == 0 or lk == numProjections-1:
                dAzimutalAngle[lk-reco['selectiveProjections'][0]] = 0
            else:
                dAzimutalAngle[lk-reco['selectiveProjections'][0]] = (dAzimutalAngle[lk - 1] + 3.544907701811032 / np.sqrt(numProjections * (1.0 - dH * dH))) % (2.0 * np.pi)

    return dAzimutalAngle, dPolarAngle

def getKvecTheo(params: Any) -> tuple[np.ndarray, float]:
    def interp1(x, y, xi):
        """
        MATLAB-like interp1(x,y,xi,'linear','extrap')
        """

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        xi = np.asarray(xi, dtype=float)

        # Sort (MATLAB allows unsorted x)
        order = np.argsort(x)
        x = x[order]
        y = y[order]

        # Linear interpolation (NaN outside first)
        yi = np.interp(xi, x, y, left=np.nan, right=np.nan)

        # Left extrapolation
        m_left = (y[1] - y[0]) / (x[1] - x[0])
        left_mask = xi < x[0]
        yi[left_mask] = y[0] + m_left * (xi[left_mask] - x[0])

        # Right extrapolation
        m_right = (y[-1] - y[-2]) / (x[-1] - x[-2])
        right_mask = xi > x[-1]
        yi[right_mask] = y[-1] + m_right * (xi[right_mask] - x[-1])

        return yi
    

    GRAD_RASTER_TIME = 10  # [µs]
    echo = 0
    gyromagneticRatio = params['acquisition']['larmorConstant'][echo] / 1e6  # in MHz/T
    if gyromagneticRatio == 0:
        gyromagneticRatio = params['acquisition']['larmorConstant'][0] / 1e6
    
    dwelltime = params['acquisition']['dwelltime'][echo] * 1e3  # [ns]
    max_slewrate = params['acquisition']['slewrate'][echo]  # [(T/m)/s]
    max_gradientAmplitude = params['acquisition']['gradAmp'][echo]  # [mT/m]
    t0 = params['acquisition']['t0'][echo]  # [ms]

    radialADCSamples = params['acquisition']['radialSamples']
    durationReadout = radialADCSamples * dwelltime / 1e6  # [ms]

    # CALCULATE K-SPACE VALUES

    durationRampUp = max_gradientAmplitude / max_slewrate

    # %Calculate k-space values after ramp and t0
    kRampUp = 0.5 * durationRampUp * gyromagneticRatio * max_gradientAmplitude
    k0 = kRampUp + gyromagneticRatio * max_gradientAmplitude * (t0 - durationRampUp)

    samplesRampUp = int(np.ceil(durationRampUp * 1e3 / GRAD_RASTER_TIME))
    samplest0 = int(np.ceil(t0 * 1e3 / GRAD_RASTER_TIME))
    samplesReadout = int(np.ceil(durationReadout * 1e3 / GRAD_RASTER_TIME))

    gradientAmplitudeRampDown = (k0 ** 2) * max_gradientAmplitude * ((k0 ** 3) + 3.0 * gyromagneticRatio * (k0 ** 2) * max_gradientAmplitude * (((samplesReadout - 1) * GRAD_RASTER_TIME + GRAD_RASTER_TIME / 2.0) / 1e3 - t0)) ** (-2.0 / 3.0)
    durationRampDown = gradientAmplitudeRampDown / max_slewrate

    samplesRampDown = int(np.floor(np.ceil(durationRampDown * 1e3 / GRAD_RASTER_TIME)))
    samplesGradient = samplesReadout + samplesRampDown

    # Calculate gradient strength
    gradientAmplitude_norm = np.zeros(samplesGradient, dtype=float)

    for samplesIndex in range(samplesRampUp):
        gradientAmplitude_norm[samplesIndex] = ( max_slewrate * ( samplesIndex * GRAD_RASTER_TIME + GRAD_RASTER_TIME / 2.0 ) / 1000.0 )

    for samplesIndex in range(samplesRampUp, samplest0):
        gradientAmplitude_norm[samplesIndex] = max_gradientAmplitude

    for samplesIndex in range(samplest0, samplesReadout):
        gradientAmplitude_norm[samplesIndex] = ( k0 ** 2 ) * max_gradientAmplitude * ( ( ( k0 ** 3 ) + 3.0 * gyromagneticRatio * ( k0 ** 2 ) * max_gradientAmplitude * ( ( ( samplesIndex * GRAD_RASTER_TIME + GRAD_RASTER_TIME / 2.0 ) / 1e3 ) - t0 ) ) ** ( -2.0 / 3.0 ) )

    for samplesIndex in range(samplesReadout, samplesGradient):
        gradientAmplitude_norm[samplesIndex] = ( max_gradientAmplitude - max_slewrate * ( ( samplesIndex - samplesReadout ) * GRAD_RASTER_TIME + GRAD_RASTER_TIME / 2.0 ) / 1000.0 )

    # Normalize and restrict to [0.0 , 1.0]
    for samplesIndex in range(samplesGradient):
        gradientAmplitude_norm[samplesIndex] = ( max( 0.0 , min( gradientAmplitude_norm[samplesIndex] / max_gradientAmplitude , 1.0 ) ) )
        
    # Calculate kmax
    m_dkmax = 0.0
    for samplesIndex in range(samplesReadout):
        m_dkmax += gradientAmplitude_norm[samplesIndex] * max_gradientAmplitude * (GRAD_RASTER_TIME / 1e3) * gyromagneticRatio
    
    # Delay Option

    RO_Grad_Delay_Recon = params['reconstruction']['grad_delay']  # [µs]

    # Calculate k-space values

    gradientAmplitude = gradientAmplitude_norm * max_gradientAmplitude  # [mT/m]
    gradientAmplitude = gradientAmplitude[0 : (samplesReadout + int( -RO_Grad_Delay_Recon / GRAD_RASTER_TIME ))]  # [mT/m]
    gradientAmplitude_Samples = len(gradientAmplitude)  # [#]
    gradientAmplitude_Time = np.arange(GRAD_RASTER_TIME / 1e3 / 2, gradientAmplitude_Samples * (GRAD_RASTER_TIME / 1e3), GRAD_RASTER_TIME / 1e3)  # [ms]

    gradientAmplitude_Interp_Time_Raster = np.gcd( int(RO_Grad_Delay_Recon * 1e3), int(np.gcd( int(GRAD_RASTER_TIME / 2 * 1e3) , int(dwelltime / 2) )) ) / 1e6  # [ms]
    gradientAmplitude_Interp_Samples = int( gradientAmplitude_Samples * ( (GRAD_RASTER_TIME / 1e3) / gradientAmplitude_Interp_Time_Raster ) )  # [#]
    gradientAmplitude_Interp_Time = np.arange( gradientAmplitude_Interp_Time_Raster / 2, gradientAmplitude_Interp_Samples * gradientAmplitude_Interp_Time_Raster, gradientAmplitude_Interp_Time_Raster )  # [ms]
    gradientAmplitude_Interp = interp1(gradientAmplitude_Time, gradientAmplitude, gradientAmplitude_Interp_Time)
    kvec_Interp_Time = (RO_Grad_Delay_Recon / 1e3) + np.arange( gradientAmplitude_Interp_Time_Raster , ( len(gradientAmplitude_Interp) + 1 ) * gradientAmplitude_Interp_Time_Raster , gradientAmplitude_Interp_Time_Raster )  # [ms]
    kvec_Interp = np.cumsum( gradientAmplitude_Interp * gradientAmplitude_Interp_Time_Raster * gyromagneticRatio )  # [1/m]
    adc_Time = np.arange( dwelltime / 1e6 / 2 , ( radialADCSamples * ( dwelltime / 1e6 ) ) , dwelltime / 1e6 )  # [ms]

    kvec = kvec_Interp[ np.isin( np.round( kvec_Interp_Time , 9 ) , np.round( adc_Time , 9 ) ) ]  # [1/m]

    kvec_axis = np.zeros((3, len(kvec)), dtype=float)
    kvec_axis[0, :] = kvec
    kvec_axis[1, :] = kvec
    kvec_axis[2, :] = kvec

    resolution = 1 / (2 * kvec[radialADCSamples-1]) * 1000  # [mm] 

    return kvec_axis, resolution

def getCoordinates(azimutalAngle, polarAngle, kvec_in, version=1):
    """
    Mimic MATLAB:
      function [kvec] = getCoordinates(azimutalAngle, polarAngle, kvec_in, version)

    Inputs
    - azimutalAngle: (N,) angles (radians)
    - polarAngle:    (N,) angles (radians)
    - kvec_in:       (3, radialSamples)
    - version:       1 or 2

    Output
    - kvec:          (3, radialSamples * N)
    """
    azimutalAngle = np.asarray(azimutalAngle, dtype=float).ravel()
    polarAngle = np.asarray(polarAngle, dtype=float).ravel()
    kvec_in = np.asarray(kvec_in, dtype=float)

    if kvec_in.shape[0] != 3:
        raise ValueError(f"kvec_in must have shape (3, radialSamples). Got {kvec_in.shape}")

    radialSamples = kvec_in.shape[1]
    numberProjections = azimutalAngle.size

    if polarAngle.size != numberProjections:
        raise ValueError("azimutalAngle and polarAngle must have the same length")

    if version == 1:
        # Preallocate: 3 x radialSamples x numberProjections
        kvec_coords = np.zeros((3, radialSamples, numberProjections), dtype=float)

        for lk in range(numberProjections):
            sinP = np.sin(polarAngle[lk])
            cosP = np.cos(polarAngle[lk])
            sinA = np.sin(azimutalAngle[lk])
            cosA = np.cos(azimutalAngle[lk])

            kvec_coords[0, :, lk] = sinA * sinP * kvec_in[0, :]
            kvec_coords[1, :, lk] = cosA * sinP * kvec_in[1, :]
            kvec_coords[2, :, lk] = cosP * kvec_in[2, :]

        # MATLAB reshape is column-major; use order='F' to match exactly
        kvec = np.zeros((3, radialSamples * numberProjections), dtype=float)
        kvec[0, :] = kvec_coords[0, :, :].reshape(radialSamples * numberProjections, order="F")
        kvec[1, :] = kvec_coords[1, :, :].reshape(radialSamples * numberProjections, order="F")
        kvec[2, :] = kvec_coords[2, :, :].reshape(radialSamples * numberProjections, order="F")
        return kvec

    elif version == 2:
        adK_ana_x = np.zeros((radialSamples, numberProjections), dtype=float)
        adK_ana_y = np.zeros((radialSamples, numberProjections), dtype=float)
        adK_ana_z = np.zeros((radialSamples, numberProjections), dtype=float)

        for lk in range(numberProjections):
            r31 = np.cos(azimutalAngle[lk]) * np.sin(polarAngle[lk])
            r32 = np.sin(azimutalAngle[lk]) * np.sin(polarAngle[lk])
            r33 = np.cos(polarAngle[lk])

            adK_ana_x[:, lk] = r31 * kvec_in[0, :]
            adK_ana_y[:, lk] = r32 * kvec_in[1, :]
            adK_ana_z[:, lk] = r33 * kvec_in[2, :]

        kvec = np.zeros((3, radialSamples * numberProjections), dtype=float)
        kvec[0, :] = adK_ana_x.reshape(radialSamples * numberProjections, order="F")
        kvec[1, :] = -adK_ana_y.reshape(radialSamples * numberProjections, order="F")  # note the minus!
        kvec[2, :] = adK_ana_z.reshape(radialSamples * numberProjections, order="F")
        return kvec

    else:
        raise ValueError("Calculation of Coordinates not known for this Sequence")

def getKvecRotation(kvec, rotation):
    """
    Mimics MATLAB getKvecRotation
    """

    def rotationmat3D(r, axis):
        """
        Create 3D rotation matrix (Rodrigues formula)
        Equivalent to MATLAB rotationmat3D
        """

        axis = np.asarray(axis, dtype=float)

        L = np.linalg.norm(axis)
        if L < np.finfo(float).eps:
            raise ValueError("axis direction must be non-zero vector")

        axis = axis / L

        u, v, w = axis
        u2, v2, w2 = u*u, v*v, w*w

        c = np.cos(r)
        s = np.sin(r)

        R = np.empty((3, 3))

        R[0, 0] = u2 + (v2 + w2) * c
        R[0, 1] = u*v*(1 - c) - w*s
        R[0, 2] = u*w*(1 - c) + v*s

        R[1, 0] = u*v*(1 - c) + w*s
        R[1, 1] = v2 + (u2 + w2) * c
        R[1, 2] = v*w*(1 - c) - u*s

        R[2, 0] = u*w*(1 - c) - v*s
        R[2, 1] = v*w*(1 - c) + u*s
        R[2, 2] = w2 + (u2 + v2) * c

        return R

    kvec = np.asarray(kvec, dtype=float)
    rotation = np.asarray(rotation, dtype=float)

    # Degrees → radians
    rx = rotation[0] / 180 * np.pi
    ry = rotation[1] / 180 * np.pi
    rz = rotation[2] / 180 * np.pi

    # Rotation matrices
    rotmatx = rotationmat3D(rx, [1, 0, 0])
    rotmaty = rotationmat3D(ry, [0, 1, 0])
    rotmatz = rotationmat3D(rz, [0, 0, 1])

    # Apply in same order as MATLAB
    kvec_rot = kvec.copy()

    kvec_rot = rotmatx @ kvec_rot
    kvec_rot = rotmaty @ kvec_rot
    kvec_rot = rotmatz @ kvec_rot

    return kvec_rot

def getSpatialDataTransform(data, kvec, params, shift_ppm):
    """
    Python mimic of MATLAB:
        [data_trans, params] = getSpatialDataTransform(data, kvec, params, iteration)

    Parameters
    ----------
    data : array_like (L, K), complex
    kvec : array_like, length 3*L*K (or shape compatible), real/complex
        Flattened k-space vector that will be reshaped to (3, L, K).
    params : object or dict-like
        Must provide:
          params.reconstruction.translation -> (3,) in "percent cm" units per MATLAB (multiplied by 0.01)
          params.offresonant.shift -> indexable by iteration
          params.acquisition.frequency
        And will be updated if off-resonance is enabled:
          params.offresonant.omega_off
          params.offresonant.TE_array
    iteration : int
        Index used for params.offresonant.shift(iteration) in MATLAB.
        If your shift is 1-based in MATLAB, pass iteration-1 here or adjust accordingly.
    getTEArray_Rastered : callable
        Function handle: TE_array = getTEArray_Rastered(params)

    Returns
    -------
    data_trans : ndarray (L, K), complex
    params : same object, possibly updated with omega_off and TE_array
    """

    data = np.asarray(data)
    datasize = data.shape  # (L, K)

    L, K = data.shape[-2:]

    # --- translations (MATLAB multiplies by 0.01) ---
    # Supports both dict-like and attribute-like params
    def _get(obj, name):
        return obj[name] if isinstance(obj, dict) else getattr(obj, name)

    def _set(obj, name, value):
        if isinstance(obj, dict):
            obj[name] = value
        else:
            setattr(obj, name, value)

    reconstruction = _get(params, "reconstruction")
    translation = _get(reconstruction, "translation")
    transx = translation[0] * 0.01
    transy = translation[1] * 0.01
    transz = translation[2] * 0.01

    # --- reshape kvec to (3, L, K) like MATLAB reshape(kvec,3,L,K) ---
    # MATLAB is column-major; if your kvec came from MATLAB, use order='F'.
    kvec_arr = np.asarray(kvec)
    if kvec_arr.size != 3 * L * K:
        raise ValueError(
            f"`kvec` must have {3*L*K} elements for reshape to (3,{L},{K}); got {kvec_arr.size}."
        )
    kvec_arr = kvec_arr.reshape((3, L,K))
    # --- optional off-resonance setup ---
    offres = _get(params, "offresonant")

    # shift may be a list/np array or callable in MATLAB style; handle both
    if shift_ppm != 0:
        acquisition = _get(params, "acquisition")
        freq = params['acquisition']['frequency'][0]
        omega_off = freq * shift_ppm * 1e-6  # Hz
        _set(offres, "omega_off", omega_off)

        TE_array = getTEArray_Rastered(params)
        _set(offres, "TE_array", TE_array)
    else:
        omega_off = None
        TE_array = None

    # --- main phase term: exp(1i*2*pi*(kx*tx + ky*ty + kz*tz)) ---
    spatial_phase = (kvec_arr[0] * transx +
                     kvec_arr[1] * transy +
                     kvec_arr[2] * transz)
    data_trans = data * np.exp(1j * 2 * np.pi * spatial_phase)
    # --- optional off-resonance phase term ---
    if shift_ppm != 0:
        TE_array = np.asarray(TE_array).T
        if TE_array.shape != (L, K):
            raise ValueError(f"TE_array must have shape {(K, L)}; got {TE_array.shape}.")
        phase_offset = np.exp(1j * 2 * np.pi * (omega_off * TE_array * 1e-6))
        data_trans = data_trans * phase_offset

    return data_trans, params

def getTEArray_Rastered(params):

    acq = params['acquisition']
    rec = params['reconstruction']

    TEIncrement = acq['echospacing'] / acq['projections']

    n_base = acq['projections'] * acq['echos']
    base_TE0 = acq['TE'][0] * 1e3

    i = np.arange(n_base)

    rawTE = base_TE0 + i * TEIncrement

    BaseTEs = np.ceil(np.floor(rawTE) / 10) * 10

    start = rec['selectiveProjections'][0]
    end = rec['selectiveProjections'][1]

    NumProjections = end - start
    BaseResolution = acq['radialSamples']

    DwellTime = acq['dwelltime'][0]

    selected_base = BaseTEs[start:end]

    l = np.arange(BaseResolution).reshape(-1, 1)

    TEArray = selected_base + l * DwellTime

    return TEArray

def getTEArray_Rastered_pyth(echospacing, projections, echos, TE0, selectiveProjections, radialSamples, dwelltime):


    TEIncrement = echospacing / projections

    n_base = projections * echos 
    base_TE0 = TE0 * 1e3

    i = np.arange(n_base)

    rawTE = base_TE0 + i * TEIncrement

    BaseTEs = np.ceil(np.floor(rawTE) / 10) * 10

    start = selectiveProjections[0]
    end = selectiveProjections[1]

    NumProjections = end - start
    BaseResolution = radialSamples

    DwellTime = dwelltime

    selected_base = BaseTEs[start:end]
    selected_base = selected_base.reshape(-1, 1)

    l = np.arange(BaseResolution)
    l = l.reshape(1, -1)

    TEArray = l * DwellTime + selected_base

    return TEArray

def getDensityCompensation(data, kvec_Coord):

    data = np.asarray(data)

    numberProjections, radialSamples = data.shape[-2:]

    # k-space magnitude
    kx = kvec_Coord[0, :radialSamples]
    ky = kvec_Coord[1, :radialSamples]
    kz = kvec_Coord[2, :radialSamples]

    kvec = np.sqrt(kx**2 + ky**2 + kz**2).squeeze()

    densityCompensation = np.zeros(radialSamples, dtype=float)

    # First
    densityCompensation[0] = (
        np.pi * (kvec[0] + kvec[1])**3
    ) / (6 * numberProjections)

    # Middle
    densityCompensation[1:-1] = (
        np.pi *
        (
            (kvec[1:-1] + kvec[2:])**3
            - (kvec[1:-1] + kvec[:-2])**3
        )
    ) / (6 * numberProjections)

    # Last
    densityCompensation[-1] = densityCompensation[-2]

    # Normalize
    densityCompensation /= np.max(densityCompensation)

    # Apply (broadcasting)
    data = data * densityCompensation[None, :]

    return data, densityCompensation

def getFilter(data, kvec_Coord, params):

    data_shape = data.shape

    filteredData = np.zeros_like(data)

    kvec = np.squeeze(np.sqrt(kvec_Coord[0,:data_shape[-1]]**2 + kvec_Coord[1,:data_shape[-1]]**2 + kvec_Coord[2,:data_shape[-1]]**2))  # (3, radialSamples)

    kvec_max = np.max(kvec)

    if params['reconstruction']['filter']['type'] == 0:  # 0 = no filter
        filter = np.ones(data_shape[-1], dtype=np.complex64)

    elif params['reconstruction']['filter']['type'] == 1:  # 1 = Hamming filter
        W = 2 * kvec_max
        a = np.pi / W
        alpha = 25 / 46
        filter = alpha + (1 - alpha) * np.cos(2 * a * kvec)
    elif params['reconstruction']['filter']['type'] == 2:  # 2 = Hanning filter
        W = 2 * kvec_max
        a = np.pi / W
        alpha = 25 / 50
        filter = alpha + (1 - alpha) * np.cos(2 * a * kvec)
    elif params['reconstruction']['filter']['type'] == 3:  # 3 = Gauss filter
        filter = np.exp(-2 * np.pi**2 * kvec**2 * params['reconstruction']['filter']['sigma']**2)

    for i in range(data_shape[-1]):  # radial samples
        filteredData[..., i] = data[..., i] * filter[i]

    return filteredData, filter

def getCoilCombine(im1: np.ndarray, filtsize: int = 7) -> np.ndarray:
    from scipy.signal import convolve2d

    """
    Python/Numpy implementation of the provided MATLAB code.

    Parameters
    ----------
    im1 : np.ndarray
        Input array with shape (sx, sy, N, C).
        Can be real or complex, e.g. complex64 / complex128.
    filtsize : int
        Square window size used for local correlation smoothing (default 7).

    Returns
    -------
    im2 : np.ndarray
        Output combined image with shape (sx, sy, 1, 1, N), dtype matching im1.
    """
    im1 = im1[..., np.newaxis] if im1.ndim == 3 else im1  # add channel dim if missing
    im1 = np.transpose(im1, (0,3,1,2))
    im1 = im1.T

    if im1.ndim != 4:
        raise ValueError("im1 must have shape (sx, sy, N, C)")

    sx, sy, N, C = im1.shape
    kernel = np.ones((filtsize, filtsize), dtype=im1.dtype)

    # --- Step 1: Compute local correlation matrices Rs
    # Rs: (sx, sy, C, C)
    Rs = np.zeros((sx, sy, C, C), dtype=im1.dtype)

    for kc1 in range(C):
        for kc2 in range(kc1, C):  # symmetry: compute only half
            # tmp(x,y) = sum_n im1(x,y,n,kc1) * conj(im1(x,y,n,kc2))
            tmp = np.sum(im1[:, :, :, kc1] * np.conj(im1[:, :, :, kc2]), axis=2)

            # MATLAB filter2(kernel, tmp, 'same') is 2D convolution with zero padding, same output size
            tmp_f = convolve2d(tmp, kernel, mode="same", boundary="fill", fillvalue=0)

            Rs[:, :, kc1, kc2] = tmp_f
            if kc1 != kc2:
                Rs[:, :, kc2, kc1] = np.conj(tmp_f)

    # --- Step 2: Compute and apply coil combination filter
    # MATLAB: im2 = zeros(sx, sy, 1, 1, N, 'like', im1)
    im2 = np.zeros((sx, sy, N), dtype=im1.dtype)

    # Iterate over all voxels (matches MATLAB's linear indexing loop)
    for kx in range(sx):
        for ky in range(sy):
            Rlocal = Rs[kx, ky, :, :]  # (C, C)

            if np.isnan(Rlocal).any() or np.all(Rlocal == 0):
                continue

            # SVD for dominant coil sensitivity
            # MATLAB: [U, ~] = svd(Rlocal, 'econ'); myfilt = U(:,1);
            U, _, _ = np.linalg.svd(Rlocal, full_matrices=False)
            myfilt = U[:, 0]  # (C,)

            # MATLAB: voxelData = squeeze(im1(kx, ky, :, :)).';  % [C x N]
            voxelData = im1[kx, ky, :, :].T  # (C, N)

            # MATLAB myfilt' is conjugate-transpose -> use conj(myfilt) @ voxelData
            im2[kx, ky, :] = np.conj(myfilt) @ voxelData

    # MATLAB final cast for single
    if im1.dtype == np.float32:
        im2 = im2.astype(np.float32, copy=False)
    elif im1.dtype == np.complex64:
        im2 = im2.astype(np.complex64, copy=False)

    return im2.transpose(1,0,2)