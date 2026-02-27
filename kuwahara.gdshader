shader_type canvas_item;
render_mode unshaded;

uniform sampler2D _MainTex : hint_screen_texture;

uniform float mix_strength : hint_range(0.0, 1.0) = 1.0;

// Controls kernel diameter (try 8..16 to REALLY see it)
uniform int _KernelSize = 10;
const int _N = 8;

// Controls high frequency detail
uniform float _Hardness = 8.0;
// Controls sharpness of "paint splotches"
uniform float _Sharpness = 8.0;
// Controls polynomial weights distribution, high numbers equivalent to blurring image
uniform float _Zeta = 0.1;

// Controls kernel threshold
const float _ZeroCrossing = 0.58;
const float _Alpha = 1.0;

float gaussian(float sigma, float pos) {
    return (1.0 / sqrt(2.0 * PI * sigma * sigma)) * exp(-(pos * pos) / (2.0 * sigma * sigma));
}

vec4 sobel(vec2 d, vec2 uv) {
    vec3 sobel_x =
        ( 1.0 * texture(_MainTex, uv + vec2(-d.x, -d.y)).rgb +
          2.0 * texture(_MainTex, uv + vec2(-d.x,  0.0)).rgb +
          1.0 * texture(_MainTex, uv + vec2(-d.x,  d.y)).rgb +
         -1.0 * texture(_MainTex, uv + vec2( d.x, -d.y)).rgb +
         -2.0 * texture(_MainTex, uv + vec2( d.x,  0.0)).rgb +
         -1.0 * texture(_MainTex, uv + vec2( d.x,  d.y)).rgb) / 4.0;

    vec3 sobel_y =
        ( 1.0 * texture(_MainTex, uv + vec2(-d.x, -d.y)).rgb +
          2.0 * texture(_MainTex, uv + vec2( 0.0, -d.y)).rgb +
          1.0 * texture(_MainTex, uv + vec2( d.x, -d.y)).rgb +
         -1.0 * texture(_MainTex, uv + vec2(-d.x,  d.y)).rgb +
         -2.0 * texture(_MainTex, uv + vec2( 0.0,  d.y)).rgb +
         -1.0 * texture(_MainTex, uv + vec2( d.x,  d.y)).rgb) / 4.0;

    return vec4(dot(sobel_x, sobel_x), dot(sobel_y, sobel_y), dot(sobel_x, sobel_y), 1.0);
}

vec4 blur_tensor(vec4 tensor, vec2 uv, vec2 d) {
    int kernelRadius = 5;

    vec4 col = vec4(0.0);
    float kernelSum = 0.0;

    for (int x = -kernelRadius; x <= kernelRadius; x++) {
        float gauss = gaussian(2.0, float(x));
        col += tensor * texture(_MainTex, uv + vec2(float(x), 0.0) * d) * gauss;
        kernelSum += gauss;
    }
    col /= kernelSum;

    vec4 col2 = vec4(0.0);
    float kernelSum2 = 0.0;
    for (int y = -kernelRadius; y <= kernelRadius; y++) {
        float gauss = gaussian(2.0, float(y));
        col2 += tensor * texture(_MainTex, uv + vec2(0.0, float(y)) * d) * gauss;
        kernelSum2 += gauss;
    }

    return col2 / kernelSum2;
}

vec4 scaling_factor(vec4 t) {
    float lambda1 = 0.5 * (t.x + t.y + sqrt(t.x * t.x - 2.0 * t.x * t.y + t.y * t.y + 4.0 * t.z * t.z));
    float lambda2 = 0.5 * (t.x + t.y - sqrt(t.x * t.x - 2.0 * t.x * t.y + t.y * t.y + 4.0 * t.z * t.z));

    vec2 v = vec2(lambda1 - t.x, -t.z);
    vec2 n = (length(v) > 0.0) ? normalize(v) : vec2(0.0, 1.0);

    float phi = -atan(n.y, n.x);

    float A = 0.0;
    if (lambda1 + lambda2 > 0.0) {
        A = (lambda1 - lambda2) / (lambda1 + lambda2);
    }

    return vec4(n, phi, A);
}

vec4 kuwahara(vec2 n, float phi, float A, vec2 d, vec2 uv) {
    int radius = _KernelSize / 2;
    float a = float(radius) * clamp((_Alpha + A) / _Alpha, 0.1, 2.0);
    float b = float(radius) * clamp(_Alpha / (_Alpha + A), 0.1, 2.0);

    float cos_phi = cos(phi);
    float sin_phi = sin(phi);

    mat2 R = mat2(vec2(cos_phi, -sin_phi), vec2(sin_phi, cos_phi));
    mat2 S = mat2(vec2(0.5 / a, 0.0), vec2(0.0, 0.5 / b));
    mat2 SR = matrixCompMult(S, R);

    int max_x = int(sqrt(a * a * cos_phi * cos_phi + b * b * sin_phi * sin_phi));
    int max_y = int(sqrt(a * a * sin_phi * sin_phi + b * b * cos_phi * cos_phi));

    float sinZeroCross = sin(_ZeroCrossing);
    float eta = (_Zeta + cos(_ZeroCrossing)) / (sinZeroCross * sinZeroCross);

    vec4 m[_N];
    vec3 s[_N];
    for (int k = 0; k < _N; k++) {
        m[k] = vec4(0.0);
        s[k] = vec3(0.0);
    }

    for (int y = -max_y; y <= max_y; y++) {
        for (int x = -max_x; x <= max_x; x++) {
            vec2 vv = SR * vec2(float(x), float(y));
            if (dot(vv, vv) <= 0.25) {
                vec3 c = clamp(texture(_MainTex, uv + vec2(float(x), float(y)) * d).rgb, 0.0, 1.0);

                float sum = 0.0;
                float w[_N];

                float vxx = _Zeta - eta * vv.x * vv.x;
                float vyy = _Zeta - eta * vv.y * vv.y;

                float z;
                z = max(0.0, vv.y + vxx); w[0] = z*z; sum += w[0];
                z = max(0.0, -vv.x + vyy); w[2] = z*z; sum += w[2];
                z = max(0.0, -vv.y + vxx); w[4] = z*z; sum += w[4];
                z = max(0.0, vv.x + vyy); w[6] = z*z; sum += w[6];

                vec2 vv2 = sqrt(2.0) * 0.5 * vec2(vv.x - vv.y, vv.x + vv.y);
                vxx = _Zeta - eta * vv2.x * vv2.x;
                vyy = _Zeta - eta * vv2.y * vv2.y;

                z = max(0.0, vv2.y + vxx); w[1] = z*z; sum += w[1];
                z = max(0.0, -vv2.x + vyy); w[3] = z*z; sum += w[3];
                z = max(0.0, -vv2.y + vxx); w[5] = z*z; sum += w[5];
                z = max(0.0, vv2.x + vyy); w[7] = z*z; sum += w[7];

                float g = exp(-3.125 * dot(vv2, vv2)) / max(sum, 1e-6);

                for (int k = 0; k < _N; k++) {
                    float wk = w[k] * g;
                    m[k] += vec4(c * wk, wk);
                    s[k] += c * c * wk;
                }
            }
        }
    }

    vec4 outc = vec4(0.0);
    for (int k = 0; k < _N; k++) {
        vec3 mean = m[k].rgb / max(m[k].w, 1e-6);
        vec3 varc = abs(s[k] / max(m[k].w, 1e-6) - mean * mean);

        float sigma2 = varc.r + varc.g + varc.b;
        float ww = 1.0 / (1.0 + pow(_Hardness * 1000.0 * sigma2, 0.5 * _Sharpness));

        outc += vec4(mean * ww, ww);
    }

    return clamp(outc / max(outc.w, 1e-6), 0.0, 1.0);
}

void fragment() {
    vec2 uv = SCREEN_UV;
    vec2 d = SCREEN_PIXEL_SIZE;

    vec4 tensor = sobel(d, uv);
    vec4 fac = scaling_factor(tensor);
    vec4 fac_blur = blur_tensor(fac, uv, d);
    vec4 filtered = kuwahara(fac_blur.xy, fac_blur.z, fac_blur.w, d, uv);

    vec4 original = texture(_MainTex, uv);
    COLOR = mix(original, filtered, mix_strength);
}