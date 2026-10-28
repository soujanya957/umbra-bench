import type { Metadata } from "next";
import { Cinzel } from "next/font/google";
import "./globals.css";

// display face for the UMBRA wordmark — Roman inscriptional capitals
const cinzel = Cinzel({
  subsets: ["latin"],
  weight: ["600", "700"],
  variable: "--font-cinzel",
  display: "swap",
});

export const metadata: Metadata = {
  title: "UMBRA",
  description:
    "UMBRA — multi-robot shadow art. Choreographing robot arms to cast designed shadows.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={cinzel.variable}>
      <body className="min-h-screen bg-umbra-bg font-sans antialiased">{children}</body>
    </html>
  );
}
