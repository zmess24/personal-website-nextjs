import { Html, Head, Main, NextScript } from "next/document";

// Runs before the page paints, so a saved theme never flashes the other one first.
const THEME_SCRIPT = `try{var t=localStorage.getItem("theme");if(t==="light"||t==="dark")document.documentElement.dataset.theme=t}catch(e){}`;

export default function Document() {
	return (
		<Html lang="en">
			<Head>
				<meta name="theme-color" content="#f4f4f2" media="(prefers-color-scheme: light)" />
				<meta name="theme-color" content="#0b0b0b" media="(prefers-color-scheme: dark)" />
				<script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
			</Head>
			<body>
				<Main />
				<NextScript />
			</body>
		</Html>
	);
}
