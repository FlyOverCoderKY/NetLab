import AppHeader from "./components/AppHeader";
import Footer from "./components/Footer";
import { ThemeProvider } from "./context/ThemeContext";
import "./App.css";
import { useEffect, useState } from "react";
import { initTFBackend } from "./utils/tf";
import AppToolbar from "./ui/components/AppToolbar";
import { RoutesView } from "./ui/routes";

function App() {
  const [backend, setBackend] = useState<string>("");
  useEffect(() => {
    initTFBackend()
      .then(setBackend)
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
          <AppToolbar />
          <main style={{ flex: 1 }}>
            <RoutesView />
          </main>
          <Footer backend={backend} />
        </div>
      )}
    </ThemeProvider>
  );
}

export default App;
