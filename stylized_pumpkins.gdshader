shader_type spatial;
render_mode diffuse_lambert, specular_disabled, ambient_light_disabled;

uniform vec4 albedo : source_color = vec4(0.968627, 0.439216, 0.086275, 1.0);
uniform vec4 shadow_color : source_color = vec4(0.2, 0.15, 0.4, 1.0);
uniform float wash : hint_range(0.0, 1.0) = 0.3;
uniform float bands : hint_range(2.0, 8.0, 1.0) = 3.0;
uniform sampler2D noise_pattern : filter_linear, repeat_enable;
uniform sampler2D light_noise : filter_linear, repeat_enable;
uniform float edge_noise_strength : hint_range(0.0, 0.5) = 0.1;

void fragment() {
	float noise_value = texture(noise_pattern, UV).r;
	float variation = mix(0.85, 1.0, noise_value);
	ALBEDO = albedo.rgb * variation;
}

void light() {
    float NdotL = dot(NORMAL, LIGHT) * 0.5 + 0.5;
	
	// sample noise for "rough" edge of cell shading
	float noise = texture(light_noise, UV).r * 2.0 - 1.0;
	// add noise to light value
	float noisy_light = NdotL + noise * edge_noise_strength;
	
    float light_intensity = floor(noisy_light * bands) / (bands - 1.0);
    vec3 lit = mix(ALBEDO, ALBEDO * LIGHT_COLOR, wash);
    vec3 shadowed = mix(ALBEDO, shadow_color.rgb, wash);
    DIFFUSE_LIGHT += mix(shadowed, lit, light_intensity);
}