import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
import pandas as pd
import os

#  Ts 就是 1/sampling time
Ts = 0.001

theta_zero = np.pi / 4
magnitude_zero = -0.3
a0 = magnitude_zero * np.exp(1j * theta_zero)
a1 = magnitude_zero * np.exp(-1j * theta_zero)

theta_pole = np.pi / 2
magnitude_pole = 0.8
b0 = magnitude_pole * np.exp(1j * theta_pole)  # 極點
b1 = magnitude_pole * np.exp(-1j * theta_pole) # 極點

# H(z)
num = np.polymul([1, -a0], [1, -a1]).real  
print(num)
den = np.polymul([1, -b0], [1, -b1])
den = np.real_if_close(den)

H_z = ctrl.TransferFunction(num, den, Ts)
print(H_z)
# ---------- PID  ----------=
Kp = 5
Ki = 0.2 
Kd = 0   

# 只有 P 控制器
C_z = ctrl.TransferFunction([Kp], [1], Ts)


# 只有 PI 控制器
C_num = [Kp + (Ki * Ts) / 2, Kp - (Ki * Ts) / 2]
C_den = [1, -1]
C_z = ctrl.TransferFunction(C_num, C_den, Ts)
# print(C_z)

# G_cl = C(z) H(z) / (1 + C(z) H(z))
G_cl = ctrl.feedback(C_z * H_z)
print(G_cl)

# ----------  Nyquist  ----------
plt.figure(figsize=(6, 6))
ctrl.nyquist(G_cl)
plt.title("Nyquist Plot of the Closed-loop System")

zeros = G_cl.zeros().tolist()
poles = G_cl.poles().tolist()
for zero in zeros:
    plt.plot(zero.real, zero.imag, 'go', label=f'Zero at {zero.real:.2f}+{zero.imag:.2f}j')
for pole in poles:
    plt.plot(pole.real, pole.imag, 'rx', label=f'Pole at {pole.real:.2f}+{pole.imag:.2f}j')

# unit_circle = plt.Circle((0, 0), 1, color='b', fill=False, linestyle='--')
# plt.gca().add_artist(unit_circle)

plt.legend()
plt.grid()
plt.show()

# ----------  Bode plot ----------
plt.figure()
ctrl.bode(G_cl, dB=True)
plt.show()

# ---------- input  ----------

time = np.arange(0, 0.1, Ts)

def step_input(time, step_time=0.02, amplitude=1):
    """ generate (Step) input signal """
    input_signal = np.zeros_like(time)
    input_signal[time >= step_time] = amplitude
    return input_signal

def impulse_input(time, impulse_time=0.02, amplitude=1):
    """ generate (Impulse) input signal """
    input_signal = np.zeros_like(time)
    idx = np.where(time >= impulse_time)[0][0]
    input_signal[idx] = amplitude
    return input_signal

def three_stage_ramp(time, t_peak1, t_peak2, slope_up1=1, slope_down=-1, slope_up2=0.5):
    """ generate (Ramp) input signal """
    y = np.piecewise(time,
        [time < t_peak1, 
         (time >= t_peak1) & (time < t_peak2), 
         time >= t_peak2],
        [lambda t: slope_up1 * t,  
         lambda t: slope_down * (t - t_peak1) + slope_up1 * t_peak1,  
         lambda t: slope_up2 * (t - t_peak2) + (slope_down * (t_peak2 - t_peak1) + slope_up1 * t_peak1)]
    )
    return y

def sine_wave_input(time, frequency=5, amplitude=1):
    """ generate (Sine Wave) input signal """
    return amplitude * np.sin(2 * np.pi * frequency * time)

get_input = {
    'step': step_input,
    'impulse': impulse_input,
    'three_stage_ramp': three_stage_ramp,
    'sine': sine_wave_input
}

input_signal = get_input['three_stage_ramp'](time, t_peak1=0.02, t_peak2=0.05, slope_up1=2, slope_down=-1, slope_up2=1)
time, response = ctrl.forced_response(G_cl, time, input_signal)

plt.figure(figsize=(8, 6))

# 
plt.subplot(2, 1, 1)
plt.plot(time, input_signal, label="Input (Step Function)")
plt.title("Input Step Function")
plt.xlabel("Time (s)")
plt.ylabel("Input")
plt.grid(True)

# 
plt.subplot(2, 1, 2)
plt.plot(time, response, label="Output Response", color='r')
plt.title("System Output Response")
plt.xlabel("Time (s)")
plt.ylabel("Output")
plt.grid(True)

plt.tight_layout()
plt.show()

output_folder = "dataset"
os.makedirs(output_folder, exist_ok=True)

data_nums = 500
for i in range(data_nums):
    # random parameters for each input signal
    step_time = np.random.uniform(0.01, 0.05)
    step_amplitude = np.random.uniform(1, 10)

    freq = np.random.uniform(1, 20)
    sine_amplitude = np.random.uniform(1, 5)

    t_peak1 = np.random.uniform(0.01, 0.03)
    t_peak2 = np.random.uniform(t_peak1 + 0.01, 0.06)
    slope_up1 = np.random.uniform(0.5, 3)
    slope_down = np.random.uniform(-2, -0.5)
    slope_up2 = np.random.uniform(0.5, 3)

    impulse_time = np.random.choice(time)  # 隨機選擇一個時間點產生脈衝
    impulse_amplitude = np.random.uniform(1, 5)

    # step input
    input_signal = get_input['step'](time, step_time=step_time, amplitude=step_amplitude)
    time, response = ctrl.forced_response(G_cl, time, input_signal)
    data = pd.DataFrame({"Time": time, "Input": input_signal, "Output": response})
    data.drop('Time', axis=1, inplace=True)
    data.to_csv(f"{output_folder}/filtered_system_data_step_{i}.csv", index=False)

    # Sine input
    input_signal = get_input['sine'](time, frequency=freq, amplitude=sine_amplitude)
    time, response = ctrl.forced_response(G_cl, time, input_signal)
    data = pd.DataFrame({"Time": time, "Input": input_signal, "Output": response})
    data.drop('Time', axis=1, inplace=True)
    data.to_csv(f"{output_folder}/filtered_system_data_sine_{i}.csv", index=False)

    # Ramp input
    input_signal = get_input['three_stage_ramp'](time, t_peak1, t_peak2, slope_up1, slope_down, slope_up2)
    time, response = ctrl.forced_response(G_cl, time, input_signal)
    data = pd.DataFrame({"Time": time, "Input": input_signal, "Output": response})
    data.drop('Time', axis=1, inplace=True)
    data.to_csv(f"{output_folder}/filtered_system_data_three_stage_ramp_{i}.csv", index=False)

    # Impulse input
    input_signal = get_input['impulse'](time, impulse_time=impulse_time, impulse_amplitude=impulse_amplitude)
    time, response = ctrl.forced_response(G_cl, time, input_signal)
    data = pd.DataFrame({"Time": time, "Input": input_signal, "Output": response})
    data.drop('Time', axis=1, inplace=True)
    data.to_csv(f"{output_folder}/filtered_system_data_impulse_{i}.csv", index=False)
