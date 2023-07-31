"""This module serves to run, choose parameters and display the output of AI
Algorithms into Discord text channel. Module is loaded up as a cog extension
that is represented by the AiAlgo class.
"""

import os
from pathlib import Path

import discord
from discord import File
from discord.ext import commands

from cogs.ai_algo import (
    stage_1_ai_evolution,
    stage_2_ai_pathfinding,
    stage_3_ai_forward_chain,
    stage_4_view,
)


class AiAlgo(commands.Cog):
    """Represents cog extension that allows us to run, choose parameters and
    display the out of AI Algorithms.

    All class attributes are parameters for the AI algorithms. We can display
    them into Discord text message to check them out, as well as modify them.
    But for the sake of simplicity, only the representative ones that affect
    the output the most are modifiable. Algorithms are being executed in stages
    by files that are placed in our current directory (ai_algo).

    Args:
        commands (discord.ext.commands.cog.CogMeta): class that is taken to
            create subclass - our own customized cog module

    Attributes:
        shared_fname (str):
        shared_points_amount (int):
        shared_climb (bool):
        evo_query (str):
        evo_begin_create (str):
        evo_max_runs (int):
        path_movement_type (str):
        path_algorithm (str):
        chain_fname_save_facts (str):
        chain_fname_load_facts (str):
        chain_fname_load_rules (str):
        chain_step_by_step (bool):
        chain_randomize_facts_order (bool):
        view_skip_rake (bool):
    """

    def __init__(self, bot):
        self.bot = bot

    # Parameters used for multiple stages.
    # stage_1_ai_evolution:     fname
    # stage_2_ai_pathfinding:   fname, points_amount, climb
    # stage_3_ai_forward_chain: fname, points_amount
    # stage_4_view:             fname,                climb
    shared_fname = "queried"
    shared_points_amount = 10  #
    shared_climb = False  #

    # Parameters for evolution algorithm stage
    evo_query = (
        "10x12 (1,5) (2,1) (3,4) (4,2) (6,8) (6,9)"  # harder: add (6,7)
    )
    evo_begin_create = "walls"
    evo_max_runs = 3

    # Parameters for pathfinding algorithm stage
    path_movement_type = "M"  #
    path_algorithm = "HK"

    # Parameters for forward chain algorithm stage
    chain_fname_save_facts = "facts"
    chain_fname_load_facts = "facts_init"
    chain_fname_load_rules = "rules"
    chain_step_by_step = True
    chain_randomize_facts_order = False  #

    view_skip_rake = False  #

    @commands.command(brief=shared_points_amount)
    async def change_points_amount(self, ctx, points_amount: int):
        """Amount of destination points to visit.

        Args:
            ctx (_type_): _description_
            points_amount (int): _description_
        """

        points_amount = int(points_amount)
        self.shared_points_amount = points_amount
        await ctx.send(f"points_amount changed to {points_amount}.")

    @commands.command(brief=shared_climb)
    async def change_climb(self, ctx, climb: int):
        """Determines approach for calculating the distance between two points
        that are adjacent to each other. If 1, distance is measured as
        abs(current terrain number - next terrain number),
        otherwise it is just (next terrain number)

        Args:
            ctx (_type_): _description_
            climb (int): _description_
        """

        self.shared_climb = bool(climb)
        await ctx.send(f"climb changed to {climb}.")

    @commands.command(brief=evo_query)
    async def change_query(self, ctx, query: str):
        """Contains size of the map and tuple coordinates of walls.
        Example: "10x12 (1,5) (2,1) (3,4) (4,2) (6,8) (6,9)

        Args:
            ctx (_type_): _description_
            query (str): _description_
        """

        self.evo_query = query
        await ctx.send(f"query changed to {query}.")

    @commands.command(brief=path_movement_type)
    async def change_movement_type(self, ctx, movement_type: str):
        """Determines whether we can move:
         - horizontally/vertically (Manhattan approach - "M"),
         - or both in the same time - diagonally (Diagonal approach - "D")

        Args:
            ctx (_type_): _description_
            movement_type (str): _description_
        """

        self.path_movement_type = movement_type
        await ctx.send(f"movement_type changed to {movement_type}.")

    @commands.command(brief=chain_randomize_facts_order)
    async def change_randomize_facts_order(
        self, ctx, randomize_facts_order: str
    ):
        """Shuffle order of facts that are going to be found.

        Args:
            ctx (_type_): _description_
            randomize_facts_order (str): _description_
        """

        self.chain_randomize_facts_order = (
            True if randomize_facts_order.lower() == "true" else False
        )
        msg = f"randomize_facts_order changed to {randomize_facts_order}."
        await ctx.send(msg)

    @commands.command(brief=view_skip_rake)
    async def change_skip_rake(self, ctx, skip_rake: int):
        """Determines whether the animation will also have the raking part.

        Args:
            ctx (_type_): _description_
            skip_rake (int): _description_
        """

        self.view_skip_rake = bool(skip_rake)
        await ctx.send(f"climb changed to {skip_rake}.")

    async def send_file_message(self, ctx):
        """_summary_

        Args:
            ctx (_type_): _description_
        """

        src_dir = Path(__file__).parents[0]
        gif_path = Path(f"{src_dir}/data/{self.shared_fname}.gif")
        with open(gif_path, "rb") as file:
            await ctx.message.channel.send(file=File(file))

    @commands.command(brief="Runs AI algs and shows gif animation of it.")
    async def run_ai(self, ctx):
        """_summary_

        Args:
            ctx (_type_): _description_
        """

        evo_parameters = dict(
            fname=self.shared_fname,
            begin_from=self.evo_begin_create,
            query=self.evo_query,
            max_runs=self.evo_max_runs,
            points_amount=self.shared_points_amount,
        )

        path_parameters = dict(
            fname=self.shared_fname,
            movement_type=self.path_movement_type,
            climb=self.shared_climb,
            algorithm=self.path_algorithm,
            visit_points_amount=self.shared_points_amount,
        )

        chain_parameters = dict(
            fname_save_facts=self.chain_fname_save_facts,
            fname_load_facts=self.chain_fname_load_facts,
            fname_load_rules=self.chain_fname_load_rules,
            step_by_step=self.chain_step_by_step,
            facts_amount=self.shared_points_amount,
            randomize_facts_order=self.chain_randomize_facts_order,
            fname=self.shared_fname,
        )

        view_parameters = dict(
            fname=self.shared_fname,
            skip_rake=self.view_skip_rake,
            climb=self.shared_climb,
        )

        stage_1_ai_evolution.create_maps(**evo_parameters)
        stage_2_ai_pathfinding.find_shortest_path(**path_parameters)
        stage_3_ai_forward_chain.run_production(**chain_parameters)
        stage_4_view.create_gif(**view_parameters)

        await self.send_file_message(ctx)


async def setup(bot):
    """Loads up this module (cog) into the bot that was initialized
    in the main function.

    Args:
        bot (__main__.MyBot): bot instance initialized in the main function
    """

    await bot.add_cog(
        AiAlgo(bot), guilds=[discord.Object(id=os.environ["SERVER_ID"])]
    )
