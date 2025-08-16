import AppHeader from "./components/AppHeader";
import Footer from "./components/Footer";
import { ThemeProvider } from "./context/ThemeContext";
import "./App.css";
import { useEffect, useState } from "react";
import { initTFBackend } from "./utils/tf";
import DatasetPanel from "./ui/components/DatasetPanel";

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
            title="Template Header"
            subtitle="A clean starting point with theming and layout"
          />
          <main style={{ flex: 1 }}>
            <section className="gradient-bg" style={{ padding: "2rem 1rem" }}>
              <div style={{ maxWidth: 1000, margin: "0 auto" }}>
                <h2>Welcome</h2>
                <p>Build your content here.</p>
              </div>
            </section>
            <DatasetPanel />
          </main>
          <Footer backend={backend} />
        </div>
      )}
    </ThemeProvider>
  );
}

export default App;
