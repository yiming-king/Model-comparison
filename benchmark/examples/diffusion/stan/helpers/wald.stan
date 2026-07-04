    real wald_lpdf(real y, real alpha, real nu) {
        real lpdf;

        lpdf = (
            log(alpha) - 1.0/2.0 * log(2*pi()) - 3.0/2.0*log(y) - (alpha - nu*y)^2/(2*y)
        );

        return lpdf;
    }

    real wald_lcdf(real y, real alpha, real nu) {
        real mu = alpha/nu;
        real lambda = alpha^2;

        real ly = sqrt(lambda/y);
        real ymu = y/mu;

        vector[2] terms;

        terms[1] = std_normal_lcdf(ly * (ymu - 1));
        terms[2] = 2*lambda/mu;
        terms[2] += std_normal_lcdf(- ly * (ymu + 1));

        real result = log_sum_exp(terms);

        return result;
    }

    real wald_lccdf(real y, real alpha, real nu) {
        real lcdf = wald_lcdf(y | alpha, nu);
        return log1m_exp(lcdf);
    }

    real wald_rng(real alpha, real nu) {
        real mu = alpha/nu;
        real lambda = alpha^2;
        real zeta = normal_rng(0, 1);
        real zeta_sq = zeta^2;
        real x = mu + (mu^2*zeta_sq)/(2*lambda) - mu/(2*lambda)*sqrt(4*mu*lambda*zeta_sq + mu^2*zeta_sq^2);
        real z = uniform_rng(0, 1);
        real y;

        if(z <= mu / (mu + x)){
            y = x;
        } else {
            y = mu^2/x;
        }

        return y;
    }

    real rdm_lpdf(real rt, real alpha, array[] real nu, real t0) {
        real lpdf = 0.0;
        real lccdf = 0.0;

        if (rt >= 0.0) {
          lpdf  = wald_lpdf (rt - t0 | alpha, nu[1]);
          lccdf = wald_lccdf(rt - t0 | alpha, nu[2]);
        } else {
          lpdf  = wald_lpdf (abs(rt) - t0 | alpha, nu[2]);
          lccdf = wald_lccdf(abs(rt) - t0 | alpha, nu[1]);
        }

        return lpdf + lccdf;
    }
