import React from "react";
import AppToolbar from "../components/AppToolbar";
import { RoutesView } from "../../ui/routes";

const LegacyShell: React.FC = () => {
  return (
    <div>
      <div
        style={{
          padding: 12,
          color: "var(--color-foreground-subtle)",
          fontSize: 12,
        }}
      >
        You are viewing the legacy UI (tabs). New navigation is available via
        the top nav.
      </div>
      <AppToolbar />
      <div style={{ padding: 12 }}>
        <RoutesView />
      </div>
    </div>
  );
};

export default LegacyShell;


