using System;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.Marshalling;

// Phi Silica Native Bridge
// 
// This C# library wraps the Windows App SDK Phi Silica APIs and exposes
// C-compatible functions for Rust FFI.
//
// Compiled with Native AOT to produce a standalone native library.

namespace PhiSilicaNative;

public static class Bridge
{
    // Availability codes matching Rust enum
    private const int AVAILABLE = 0;
    private const int UNAVAILABLE = 1;
    private const int WINDOWS_TOO_OLD = 2;
    private const int NPU_NOT_DETECTED = 3;

    /// <summary>
    /// Check if Phi Silica is available on this device.
    /// </summary>
    [UnmanagedCallersOnly(EntryPoint = "phi_silica_availability")]
    public static int GetAvailability()
    {
        try
        {
            // Try to access Microsoft.Windows.AI.Text.LanguageModel
            // This requires Windows App SDK 1.6+ and Windows 11 24H2+
            var readyState = GetLanguageModelReadyState();
            
            if (readyState == 1) // Ready
                return AVAILABLE;
            if (readyState == 0) // NotReady
                return NPU_NOT_DETECTED;
            
            return UNAVAILABLE;
        }
        catch (Exception)
        {
            // Windows App SDK not available or Phi Silica not supported
            return UNAVAILABLE;
        }
    }

    /// <summary>
    /// Generate text from a prompt.
    /// </summary>
    [UnmanagedCallersOnly(EntryPoint = "phi_silica_generate")]
    public static IntPtr Generate(IntPtr promptPtr, int maxTokens)
    {
        try
        {
            string? prompt = Marshal.PtrToStringUTF8(promptPtr);
            if (prompt == null)
                return Marshal.StringToCoTaskMemUTF8("[error: null prompt]");

            // Attempt to call Phi Silica via Windows App SDK
            string? result = GenerateWithPhiSilica(prompt, maxTokens);
            
            if (result != null)
                return Marshal.StringToCoTaskMemUTF8(result);
            
            // Rust treats the error prefix as a bridge failure.
            return Marshal.StringToCoTaskMemUTF8(
                "[error: Phi Silica C# bridge could not generate a response. " +
                "Use a custom PhiSilicaBridge implementation.]");
        }
        catch (Exception ex)
        {
            return Marshal.StringToCoTaskMemUTF8($"[error: {ex.Message}]");
        }
    }

    /// <summary>
    /// Free a string allocated by the bridge.
    /// </summary>
    [UnmanagedCallersOnly(EntryPoint = "phi_silica_free_string")]
    public static void FreeString(IntPtr ptr)
    {
        if (ptr != IntPtr.Zero)
            Marshal.FreeCoTaskMem(ptr);
    }

    // --- Internal helpers ---

    /// <summary>
    /// Attempts to get the LanguageModel ready state via reflection.
    /// Returns -1 if the type is not available.
    /// </summary>
    private static int GetLanguageModelReadyState()
    {
        try
        {
            var assembly = System.Reflection.Assembly.Load(
                "Microsoft.Windows.AI, Version=1.6.0.0, Culture=neutral, PublicKeyToken=null");
            var type = assembly?.GetType("Microsoft.Windows.AI.Text.LanguageModel");
            var method = type?.GetMethod("GetReadyState");
            var result = method?.Invoke(null, null);
            return result != null ? (int)result : -1;
        }
        catch
        {
            return -1;
        }
    }

    /// <summary>
    /// Attempts to generate text using Phi Silica via reflection.
    /// Returns null if not available.
    /// </summary>
    private static string? GenerateWithPhiSilica(string prompt, int maxTokens)
    {
        try
        {
            var assembly = System.Reflection.Assembly.Load(
                "Microsoft.Windows.AI, Version=1.6.0.0, Culture=neutral, PublicKeyToken=null");
            var type = assembly?.GetType("Microsoft.Windows.AI.Text.LanguageModel");
            
            if (type == null) return null;

            // Call EnsureReadyAsync
            var ensureMethod = type.GetMethod("EnsureReadyAsync");
            var ensureTask = ensureMethod?.Invoke(null, null);
            if (ensureTask != null)
            {
                var getAwaiter = ensureTask.GetType().GetMethod("GetAwaiter");
                var awaiter = getAwaiter?.Invoke(ensureTask, null);
                var getResult = awaiter?.GetType().GetMethod("GetResult");
                getResult?.Invoke(awaiter, null);
            }

            // Create LanguageModel instance
            var createMethod = type.GetMethod("CreateAsync");
            var createTask = createMethod?.Invoke(null, null);
            object? model = null;
            if (createTask != null)
            {
                var getAwaiter = createTask.GetType().GetMethod("GetAwaiter");
                var awaiter = getAwaiter?.Invoke(createTask, null);
                var getResult = awaiter?.GetType().GetMethod("GetResult");
                model = getResult?.Invoke(awaiter, null);
            }

            if (model == null) return null;

            // Call GenerateResponseAsync
            var generateMethod = type.GetMethod("GenerateResponseAsync", new[] { typeof(string) });
            var generateTask = generateMethod?.Invoke(model, new object[] { prompt });
            if (generateTask != null)
            {
                var getAwaiter = generateTask.GetType().GetMethod("GetAwaiter");
                var awaiter = getAwaiter?.Invoke(generateTask, null);
                var getResult = awaiter?.GetType().GetMethod("GetResult");
                var result = getResult?.Invoke(awaiter, null);
                
                if (result != null)
                {
                    var textProp = result.GetType().GetProperty("Text");
                    return textProp?.GetValue(result) as string;
                }
            }

            return null;
        }
        catch
        {
            return null;
        }
    }
}
