import AppHeader from "./components/AppHeader";
import Footer from "./components/Footer";
import { ThemeProvider } from "./context/ThemeContext";
import "./App.css";
import { useEffect, useState } from "react";
import { detectWasmCapabilities, initTFBackend } from "./utils/tf";
import { useAppStore } from "./state/store";
import TopNav from "./ui/components/TopNav";
import AppRouter from "./app/router";

function App() {
  const [backend, setBackend] = useState<string>("");
  useEffect(() => {
    initTFBackend()
      .then(async (b) => {
        try {
          const caps = await detectWasmCapabilities();
          // Optionally adjust defaults for mobile/no SIMD
          if (b === "wasm" && (!caps.simd || !caps.threads)) {
            // Reduce default batch size for constrained devices
            // Auto-tune default batch size via global store
            try {
              const setTraining = useAppStore.getState().setTraining;
              const current = useAppStore.getState().training;
              if ((current.batchSize ?? 0) > 16) {
                setTraining({ batchSize: 16 });
              }
            } catch {
              // ignore store access errors
            }
            setBackend(`${b} (reduced)`);
          } else {
            setBackend(b);
          }
        } catch {
          setBackend(b);
        }
      })
      .catch(() => setBackend("unknown"));
  }, []);
  return (
    <ThemeProvider>
      {(theme) => (
        <div className="App" data-theme={theme.resolvedAppearance}>
          <AppHeader
            title="Neural Network Visualizer"
            subtitle="Visualize the output of a neural network"
          />
          <TopNav />
          <main style={{ flex: 1 }}>
            <AppRouter />
          </main>
          <Footer backend={backend} />
        </div>
      )}
    </ThemeProvider>
  );
}

export default App;
