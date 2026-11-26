import moment from "moment";
import { JOBS, NOW } from "@/data/site";

// Geometry of the horizontal line (px).
export const PX = 48; // one month
export const CARD_W = 176; // event card width
export const LANE_UP = 104; // lane height above the axis (writing)
export const LANE_DN = 132; // lane height below the axis (built + work)

const DATE_FORMATS = ["MM-DD-YYYY", "MM/DD/YYYY", "MMM YYYY"];
const parse = (d) => moment(d, DATE_FORMATS);

// Build every event on the line from posts, earlier projects, work and current projects.
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
		return { ...e, y: m.year(), mo: m.month(), label: m.format("MMM YYYY") };
	});

	const startYear = Math.min(...ev.map((e) => e.y));
	const endYear = Math.max(...ev.map((e) => e.y));
	ev.forEach((e) => (e.m = (e.y - startYear) * 12 + e.mo));
	// Oldest first; within a month, built things before writing.
	ev.sort((a, b) => a.m - b.m || (a.kind === "post") - (b.kind === "post"));

	// Stack events that would overlap into lanes, separately above and below the axis.
	const ends = { up: [], dn: [] };
	ev.forEach((e) => {
		const side = e.kind === "post" ? ends.up : ends.dn;
		const x0 = (e.m + 0.5) * PX;
		let lane = side.findIndex((end) => end <= x0);
		if (lane === -1) {
			lane = side.length;
			side.push(0);
		}
		side[lane] = x0 + CARD_W + 10;
		e.lane = lane;
		e.x = Math.round(x0);
	});

	const months = (endYear - startYear + 1) * 12;
	const axis = 30 + 24 + ends.up.length * LANE_UP; // axis y
	const below = 40 + ends.dn.length * LANE_DN + 24;
	const years = [];
	for (let y = startYear; y <= endYear; y++) years.push({ year: y, x: (y - startYear) * 12 * PX });

	return { events: ev, years, width: months * PX + 300, axis, height: axis + below, startYear, endYear };
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
