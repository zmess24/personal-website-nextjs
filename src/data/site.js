// Hand-kept content for the home page. Posts come from src/posts and earlier
// projects from src/projects; everything else on the timeline lives here.

export const SITE_URL = "https://www.zacmessinger.com";

export const BIO =
	"Solutions architect at Stripe, working on agentic commerce. I learn AI by building it from scratch and writing it up, and I make apps at CLZ Labs.";

export const SOCIAL = [
	{ name: "LinkedIn", url: "https://www.linkedin.com/in/zacmessinger" },
	{ name: "GitHub", url: "https://github.com/zmess24" },
	{ name: "X", url: "https://twitter.com/zdmessinger" },
];

// Work, shown on the line below the axis with the things I built.
// Dates are "Mon YYYY".
export const JOBS = [
	{ title: "Stripe", subtitle: "Solutions Architect", date: "Aug 2024", url: "https://stripe.com", logo: "stripe" },
	{ title: "Agentic Commerce SA", subtitle: "Stripe", date: "Jul 2026", url: "https://stripe.com", logo: "stripe" },
];

// What I'm building now: these end the line and get the big cards.
export const NOW = [
	{
		name: "CLZ Labs",
		host: "clzlabs.com",
		url: "https://clzlabs.com",
		date: "Oct 2026",
		status: "Live",
		tagline: "Small, carefully made software",
		description: "An independent software company that designs, builds and runs its own consumer apps, starting with Rung.",
		since: "Founded Oct 2026",
		mark: "clz",
	},
	{
		name: "Rung",
		host: "rungfit.com",
		url: "https://rungfit.com",
		date: "Nov 2026",
		status: "Private preview",
		tagline: "Every rep. Every meal. One Rung at a time.",
		description:
			"An all-in-one fitness app. Build a training program, log workouts and meals, weigh in, and see your progress toward your daily targets.",
		since: "Private preview since Nov 2026",
		icon: "/images/rung-icon.png",
	},
];
