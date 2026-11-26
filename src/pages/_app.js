import { Inter_Tight, JetBrains_Mono } from "next/font/google";
import "@/styles/globals.css";
import "@/styles/home.css";
import "@/styles/post.css";

const sans = Inter_Tight({ subsets: ["latin"], weight: ["200", "300", "400", "500", "600"], display: "swap" });
const mono = JetBrains_Mono({ subsets: ["latin"], weight: ["400"], style: ["normal", "italic"], display: "swap" });

export default function App({ Component, pageProps }) {
	return (
		<>
			<style jsx global>{`
				:root {
					--sans: ${sans.style.fontFamily}, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
					--mono: ${mono.style.fontFamily}, ui-monospace, Menlo, Consolas, monospace;
				}
			`}</style>
			<Component {...pageProps} />
		</>
	);
}
