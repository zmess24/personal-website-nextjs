import moment from "moment";
import { JOBS, NOW } from "@/data/site";

const DATE_FORMATS = ["MM-DD-YYYY", "MM/DD/YYYY", "MMM YYYY"];
const parse = (d) => moment(d, DATE_FORMATS);

// Every event on the line from posts, earlier projects, work and current projects,
// grouped by year, newest first.
export function buildEvents({ posts, projects }) {
	const ev = [
		...posts.map((p) => ({
			kind: "post",
			title: p.title,
			subtitle: p.description,
			date: p.date,
			href: `/posts/${p.id}`,
		})),
		...projects.map((p) => ({
			kind: "proj",
			title: p.title,
			subtitle: p.description,
			date: p.date,
			href: p.link,
		})),
		...JOBS.map((j) => ({ kind: "job", title: j.title, subtitle: j.subtitle, date: j.date, href: j.url, logo: j.logo })),
		...NOW.map((n) => ({ kind: "now", title: n.name, subtitle: n.tagline, date: n.date, href: n.url, status: n.status })),
	].map((e) => {
		const m = parse(e.date);
		return { ...e, t: m.valueOf(), y: m.year(), label: m.format("MMM YYYY") };
	});

	// Newest first; on the same date, writing comes before the thing it was written alongside.
	ev.sort((a, b) => b.t - a.t || (b.kind === "post") - (a.kind === "post"));

	const years = [];
	ev.forEach(({ t, ...e }) => {
		if (!years.length || years[years.length - 1].year !== e.y) years.push({ year: e.y, events: [] });
		years[years.length - 1].events.push(e);
	});

	return { years, startYear: years[years.length - 1].year, endYear: years[0].year };
}

// Where a post sits on the line, as a fraction of the whole span (for the post page strip).
export function postStrip(posts, currentId) {
	const dates = posts.map((p) => parse(p.date));
	const startYear = Math.min(...dates.map((d) => d.year()), 2021);
	const endYear = Math.max(...dates.map((d) => d.year()), ...NOW.map((n) => parse(n.date).year()));
	const span = (endYear - startYear + 1) * 12;
	const at = (d) => (((d.year() - startYear) * 12 + d.month() + 0.5) / span) * 100;
	const years = [];
	for (let y = startYear; y <= endYear; y++) years.push({ year: y, left: (((y - startYear) * 12) / span) * 100 });
	return {
		years,
		ticks: posts.map((p, i) => ({ id: p.id, left: at(dates[i]), me: p.id === currentId })),
	};
}
