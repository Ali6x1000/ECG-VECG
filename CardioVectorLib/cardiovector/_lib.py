import wfdb
import numpy as np


def getindices(rec: wfdb.Record, sigs: list):
    return list(map(rec.sig_name.index, sigs))


def iterfy(iterable):
    if isinstance(iterable, str):
        iterable = [iterable]
    try:
        iter(iterable)
    except TypeError:
        iterable = [iterable]
    return iterable


def copy_record(r: wfdb.Record):
    assert isinstance(r, wfdb.Record)

    def c_(a): return None if a is None else np.array(a, copy=True)
    return wfdb.Record(
    p_signal=c_(r.p_signal),
    d_signal=c_(r.d_signal),
    e_p_signal=c_(r.e_p_signal),
    e_d_signal=c_(r.e_d_signal),
    record_name=r.record_name,
    n_sig=r.n_sig,
    fs=r.fs,
    counter_freq=r.counter_freq,
    base_counter=r.base_counter,
    sig_len=r.sig_len,
    base_time=r.base_time,
    base_date=r.base_date,
    file_name=r.file_name,
    fmt=r.fmt,
    samps_per_frame=r.samps_per_frame,
    skew=r.skew,
    byte_offset=r.byte_offset,
    adc_gain=r.adc_gain,
    baseline=r.baseline,
    units=r.units,
    adc_res=r.adc_res,
    adc_zero=r.adc_zero,
    init_value=r.init_value,
    checksum=r.checksum,
    block_size=r.block_size,
    sig_name=r.sig_name,
    comments=r.comments,
)



def get_analog(r: wfdb.Record) -> np.ndarray:
    return _get_adac(r.p_signal, r.dac)


def get_digital(r: wfdb.Record) -> np.ndarray:
    return _get_adac(r.d_signal, r.adc)


def _get_adac(sig: np.ndarray, get):
    if sig is None or sig.shape == ():
        return get()
    else:
        return sig


def validate_adac(record: wfdb.Record) -> (int, int, int):
    fmt = _get_uniq(record.fmt)
    adcgain = _get_uniq(record.adc_gain)
    baseline = _get_uniq(record.baseline)
    return fmt, adcgain, baseline


def _get_uniq(vals):
    s = set(vals)
    if len(s) != 1:
        raise ValueError('Could not get an unique value')

    return vals[0]
