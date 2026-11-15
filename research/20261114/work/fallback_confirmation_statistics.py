"""H2bk prespecified separate CP decisions; maze-level fixed-model joins."""
import audit as A


def absolute(successes,n=4096):
    lo,hi=A.bounds(successes,n,.05)
    return {'successes':successes,'n':n,'observed_success_pp':100*successes/n,
            'one_sided95_success_lower_pp':100*lo,'one_sided95_success_upper_pp':100*hi,
            'minimum_success_pp':99.9,'decision':'supported' if lo>=.999 else 'falsified' if hi<.999 else 'inconclusive'}


def noninferiority(gains,losses,n=4096):
    gl,gu=A.bounds(gains,n,.025);ll,lu=A.bounds(losses,n,.025);lower=100*(gl-lu);upper=100*(gu-ll)
    return {'gains':gains,'losses':losses,'n':n,'observed_net_gain_pp':100*(gains-losses)/n,
            'separately_one_sided95_net_lower_pp':lower,'separately_one_sided95_net_upper_pp':upper,
            'minimum_net_gain_pp':-.1,'decision':'supported' if lower>=-.1 else 'falsified' if upper<-.1 else 'inconclusive'}
