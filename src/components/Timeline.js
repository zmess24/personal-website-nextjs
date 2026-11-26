import Link from "next/link";
import StripeMark from "./StripeMark";

const KIND_CLASS = { post: "post", proj: "proj", now: "proj now", job: "wk" };

function EventLink({ e, children }) {
	if (e.href.startsWith("/")) return <Link href={e.href}>{children}</Link>;
	return (
		<a href={e.href} target="_blank" rel="noreferrer">
			{children}
		</a>
	);
}

// One vertical line, newest first, grouped by year. Writing, things I built and work share the line,
// told apart by their marker and date colour. The same list serves phones and desktop.
export default function Timeline({ layout }) {
	return (
		<ol className="tline" id="line" aria-label={`Timeline, ${layout.startYear} to now`}>
			{layout.years.map(({ year, events }) => [
				<li key={`y-${year}`} className="y" aria-hidden="true">
					<span>{year}</span>
				</li>,
				...events.map((e) => (
					<li key={`${e.kind}-${e.title}-${e.date}`} className={KIND_CLASS[e.kind]}>
						{e.logo === "stripe" && (
							<span className="vm">
								<StripeMark />
							</span>
						)}
						<EventLink e={e}>
							<span className="dt">
								{e.label}
								{e.status && <span className="st">{e.status}</span>}
							</span>
							<span className="tt">{e.title}</span>
							{e.subtitle && <span className="su">{e.subtitle}</span>}
						</EventLink>
					</li>
				)),
			])}
		</ol>
	);
}
