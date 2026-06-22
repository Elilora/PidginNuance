import type { Metadata } from "next";
import * as React from "react";
import { AppRouterCacheProvider } from "@mui/material-nextjs/v16-appRouter";
import { ThemeProvider } from "@mui/material/styles";
import CssBaseline from "@mui/material/CssBaseline";
import theme from "@/theme/theme";
import DashboardLayout from "@/component/DashboardLayout";
import "./globals.css";

export const metadata: Metadata = {
  title: "Pidgin Sentiment Dashboard",
  description: "Sentiment & emotion analysis dashboard for Nigerian Pidgin text",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <AppRouterCacheProvider>
          <ThemeProvider theme={theme}>
            <CssBaseline />
            <DashboardLayout>{children}</DashboardLayout>
          </ThemeProvider>
        </AppRouterCacheProvider>
      </body>
    </html>
  );
}