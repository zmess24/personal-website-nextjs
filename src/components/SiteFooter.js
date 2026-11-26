import { SOCIAL } from "@/data/site";

export default function SiteFooter() {
	return (
		<footer className="foot">
			<span className="caps" suppressHydrationWarning>
				&copy; {new Date().getFullYear()} Zachary Messinger
			</span>
			<nav className="caps" aria-label="Elsewhere">
				{SOCIAL.map((s) => (
					<a key={s.url} href={s.url} target="_blank" rel="noreferrer">
						{s.name}
					</a>
				))}
			</nav>
		</footer>
	);
}
