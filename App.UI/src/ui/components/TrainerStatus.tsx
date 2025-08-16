import React, { useEffect, useState } from "react";
import { getTrainerClient } from "../../worker/client";

const TrainerStatus: React.FC = () => {
  const [status, setStatus] = useState<string>("spawning");

  useEffect(() => {
    const client = getTrainerClient();
    const offReady = client.on("ready", () => setStatus("ready (wasm)"));
    const offErr = client.on("error", (m) => setStatus(`error: ${m}`));
    return () => {
      offReady();
      offErr();
    };
  }, []);

  return (
    <div
      aria-live="polite"
      style={{
        padding: "0.5rem 1rem",
        color: "var(--color-foreground-subtle)",
      }}
    >
      Worker: {status}
    </div>
  );
};

export default TrainerStatus;
