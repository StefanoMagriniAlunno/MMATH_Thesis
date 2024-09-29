# Load the sunspots dataset
data("sunspots")
signal <- sunspots  # Assign the sunspots data to 'signal'

# Get the length of the signal
n <- length(signal)

# Perform the Fast Fourier Transform (FFT)
fft_result <- fft(signal)

# Calculate amplitudes (normalized by dividing by the length of the signal)
amplitudes <- Mod(fft_result) / n

# Calculate phases (in radians)
# phases <- Arg(fft_result)

# Plot the original signal
par(mfrow = c(1, 2))  # Set the plotting window for one plot
plot(signal, main = "Sunspot Numbers", ylab = "Number of Sunspots", xlab = "Time")

# Create a sequence of frequencies for plotting
freqs <- seq(0, n - 1) / n

# Plot the amplitudes of the Fourier coefficients
plot(freqs, amplitudes, type = "h",
     xlab = "Frequency", ylab = "Amplitude",
     main = "Amplitudes of Fourier Coefficients")
