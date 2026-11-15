"""Independent SciPy beta-quantile check of paired gain/loss tail allocation."""
import json
import platform
import scipy
from scipy.stats import beta
import study as S


def bounds(k,n,alpha):
    return (0. if k==0 else float(beta.ppf(alpha,k,n-k+1)),1. if k==n else float(beta.ppf(1-alpha,k+1,n-k)))


def net(gain,loss,n,alpha):
    gl,gu=bounds(gain,n,alpha);ll,lu=bounds(loss,n,alpha)
    return {"gain_one_sided_lower_pp":100*gl,"gain_one_sided_upper_pp":100*gu,"loss_one_sided_lower_pp":100*ll,"loss_one_sided_upper_pp":100*lu,"net_lower_pp":100*(gl-lu),"net_upper_pp":100*(gu-ll),"alpha_per_marginal_tail":alpha}


def main():
    dest=S.OUT/"native_relaxation";audit=json.loads((dest/"audit.json").read_text());p=audit["primary"]
    one=net(p["gains"],p["losses"],p["n_unique"],.025);two=net(p["gains"],p["losses"],p["n_unique"],.0125)
    errors={"lower_pp":abs(one["net_lower_pp"]-p["one_sided95_net_gain_lower_pp"]),"upper_pp":abs(one["net_upper_pp"]-p["one_sided95_net_gain_upper_pp"])}
    assert max(errors.values())<1e-7
    assert one["net_lower_pp"]>=p["target_pp"]
    result={"status":"passed","utc":S.utc(),"n_unique":p["n_unique"],"gains":p["gains"],"losses":p["losses"],"scipy_version":scipy.__version__,"python":platform.python_version(),"one_sided95_prespecified":one,"two_sided95_descriptive":two,"reported_custom_cdf_absolute_error_pp":errors,
        "tail_allocation":"The prespecified lower bound subtracts a 97.5% one-sided loss-probability upper bound from a 97.5% one-sided gain-probability lower bound. Union bound over these TWO marginal failures gives a 95% one-sided net-gain lower bound, without assuming gains/losses independent. The upper bound analogously uses two marginal tails.",
        "joint_interval_caution":"The two individually one-sided95% endpoints are NOT a conventional jointly95% two-sided interval. With four marginal tails at alpha=.025, joint coverage is conservatively at least90%. A jointly95% two-sided conservative interval allocates alpha=.0125 to each of four tails; it is reported separately and was not the frozen primary decision rule.",
        "decision":"Frozen one-sided95% primary target is supported. The separate two-sided95% conservative lower bound does not establish the0.05pp target; do not relabel the primary bounds as a two-sided95% CI.","audit_sha256":S.sha(dest/"audit.json"),"script_sha256":S.sha(__file__)}
    S.save(dest/"scipy-statistics-verification.json",result);S.emit("native_relaxation_statistics_verified",one_sided95=one,two_sided95=two,maximum_error_pp=max(errors.values()))


if __name__=="__main__":main()
